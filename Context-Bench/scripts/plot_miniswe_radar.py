#!/usr/bin/env python3
"""
Plot radar charts for MiniSWE agent results across different languages.
Shows precision, recall, and F1 scores at file, symbol, and span levels.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


# Configuration
RESULTS_DIR = Path('/root/lh/Context-Bench/results/miniswe')
CSV_FILE = Path('/root/lh/Context-Bench/selected_500_instances.csv')
OUTPUT_FILE = Path('/root/lh/Context-Bench/results/miniswe_radar_charts.png')

MODELS = ['claude45', 'gemini', 'gpt5', 'mistral']
BENCHMARKS = ['Multi', 'Poly', 'Pro', 'Verified']
LEVELS = ['file', 'symbol', 'span']
METRICS = ['precision', 'recall', 'f1']

# Language configuration (8 languages for octagon)
LANGUAGES = ['python', 'java', 'cpp', 'typescript', 'c', 'rust', 'go', 'javascript']
LANG_DISPLAY = {
    'python': 'Python',
    'java': 'Java',
    'cpp': 'C++',
    'typescript': 'TS',
    'c': 'C',
    'rust': 'Rust',
    'go': 'Go',
    'javascript': 'JS'
}

# Model colors and styles
MODEL_COLORS = {
    'claude45': '#A8D8EA',  # Soft Blue
    'gemini': '#AAF0C4',    # Soft Green
    'gpt5': '#FFB6B9',      # Soft Pink/Red
    'mistral': '#C7ADDB'    # Soft Purple
}

MODEL_DISPLAY = {
    'claude45': 'Claude-3.5-Sonnet',
    'gemini': 'Gemini',
    'gpt5': 'GPT-4o',
    'mistral': 'Mistral'
}


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    
    This function creates a RadarAxes projection and registers it.
    """
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            self.set_theta_direction(-1)

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                            spine_type='circle',
                            path=MPath.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                  + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def load_language_mapping(csv_file):
    """Load instance_id to language mapping from CSV."""
    mapping = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = row.get('original_inst_id', '').strip()
            language = row.get('language', '').strip().lower()
            if instance_id and language:
                mapping[instance_id] = language
    print(f"Loaded {len(mapping)} instance-language mappings")
    return mapping


def calculate_f1(precision, recall):
    """Calculate F1 score, handling division by zero."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_results(results_dir, models, benchmarks, lang_mapping):
    """
    Load all JSONL results and aggregate by model and language.
    
    Returns:
        dict: {model: {language: {level: {metric: [values]}}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    total_files = 0
    loaded_files = 0
    missing_instances = 0
    
    for model in models:
        for bench in benchmarks:
            jsonl_file = results_dir / model / bench / 'all.jsonl'
            total_files += 1
            
            if not jsonl_file.exists():
                print(f"Warning: {jsonl_file} not found")
                continue
            
            loaded_files += 1
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        result = json.loads(line)
                        instance_id = result.get('instance_id', '')
                        
                        # Get language for this instance
                        language = lang_mapping.get(instance_id)
                        if not language:
                            missing_instances += 1
                            continue
                        
                        # Skip if language not in our target set
                        if language not in LANGUAGES:
                            continue
                        
                        # Extract metrics for each level
                        final = result.get('final', {})
                        for level in LEVELS:
                            level_data = final.get(level, {})
                            if not level_data:
                                continue
                            
                            precision = level_data.get('precision', 0.0)
                            recall = level_data.get('coverage', 0.0)  # coverage is recall
                            f1 = calculate_f1(precision, recall)
                            
                            data[model][language][level]['precision'].append(precision)
                            data[model][language][level]['recall'].append(recall)
                            data[model][language][level]['f1'].append(f1)
                    
                    except json.JSONDecodeError as e:
                        print(f"Error parsing {jsonl_file} line {line_num}: {e}")
                        continue
    
    print(f"\nLoaded {loaded_files}/{total_files} files")
    print(f"Missing language mapping for {missing_instances} instances")
    
    return data


def aggregate_data(data, models, languages, levels, metrics):
    """
    Aggregate data by computing mean for each (model, language, level, metric).
    
    Returns:
        dict: {level: {metric: {model: [values_per_language]}}}
    """
    aggregated = {}
    
    for level in levels:
        aggregated[level] = {}
        
        for metric in metrics:
            aggregated[level][metric] = {}
            
            for model in models:
                values = []
                for language in languages:
                    metric_values = data[model][language][level][metric]
                    if metric_values:
                        avg_value = np.mean(metric_values)
                        values.append(avg_value)
                    else:
                        # No data for this language
                        values.append(0.0)
                
                aggregated[level][metric][model] = values
    
    return aggregated


def plot_radar_charts(aggregated_data, languages, models, output_file):
    """
    Plot 9 radar charts (3 levels x 3 metrics) in a 3x3 grid.
    """
    # Create radar projection
    num_vars = len(languages)
    theta = radar_factory(num_vars, frame='polygon')
    
    # Create figure with 3x3 subplots
    fig = plt.figure(figsize=(18, 18))
    
    # Language labels for display
    labels = [LANG_DISPLAY.get(lang, lang) for lang in languages]
    
    subplot_idx = 1
    for row_idx, level in enumerate(LEVELS):
        for col_idx, metric in enumerate(METRICS):
            ax = fig.add_subplot(3, 3, subplot_idx, projection='radar')
            
            # Get data for this subplot
            model_data = aggregated_data[level][metric]
            
            # Plot each model
            for model in models:
                values = model_data[model]
                # Close the plot by appending the first value
                values_closed = values + [values[0]]
                theta_closed = np.concatenate([theta, [theta[0]]])
                
                color = MODEL_COLORS.get(model, 'gray')
                label = MODEL_DISPLAY.get(model, model)
                
                ax.plot(theta_closed, values_closed, 'o-', linewidth=2, 
                       color=color, label=label, markersize=6)
                ax.fill(theta_closed, values_closed, alpha=0.15, color=color)
            
            # Customize axes
            ax.set_varlabels(labels)
            ax.set_ylim(0, 1)
            
            # Set radial ticks
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            
            # Title
            title = f"{level.capitalize()} - {metric.capitalize()}"
            ax.set_title(title, size=14, weight='bold', pad=20)
            
            # Legend only for first subplot
            if subplot_idx == 1:
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            
            subplot_idx += 1
    
    plt.tight_layout(pad=3.0)
    
    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved radar charts to: {output_file}")
    
    return fig


def print_summary(aggregated_data, languages, models):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for level in LEVELS:
        print(f"\n{level.upper()} Level:")
        print("-" * 60)
        
        for metric in METRICS:
            print(f"\n  {metric.capitalize()}:")
            model_data = aggregated_data[level][metric]
            
            for model in models:
                values = model_data[model]
                avg = np.mean(values)
                print(f"    {MODEL_DISPLAY[model]:20s}: {avg:.4f}")


def main():
    """Main execution."""
    print("MiniSWE Radar Chart Generator")
    print("="*80)
    
    # Load language mapping
    print("\n1. Loading language mapping...")
    lang_mapping = load_language_mapping(CSV_FILE)
    
    # Load results
    print("\n2. Loading results from all models and benchmarks...")
    raw_data = load_results(RESULTS_DIR, MODELS, BENCHMARKS, lang_mapping)
    
    # Print data coverage
    print("\n3. Data coverage by model and language:")
    for model in MODELS:
        print(f"\n  {MODEL_DISPLAY[model]}:")
        for lang in LANGUAGES:
            if lang in raw_data[model]:
                count = len(raw_data[model][lang]['span']['precision'])
                print(f"    {LANG_DISPLAY[lang]:12s}: {count:3d} instances")
    
    # Aggregate data
    print("\n4. Aggregating data...")
    aggregated = aggregate_data(raw_data, MODELS, LANGUAGES, LEVELS, METRICS)
    
    # Print summary
    print_summary(aggregated, LANGUAGES, MODELS)
    
    # Plot radar charts
    print("\n5. Plotting radar charts...")
    plot_radar_charts(aggregated, LANGUAGES, MODELS, OUTPUT_FILE)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
