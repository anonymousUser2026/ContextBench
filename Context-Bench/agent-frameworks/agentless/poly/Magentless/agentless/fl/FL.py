from abc import ABC, abstractmethod

from agentless.multilang.const import (
    LANG_EXT,
    LANGUAGE,
)
from agentless.multilang.utils import end_with_ext
from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

MAX_CONTEXT_LENGTH = 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.{lang_ext}
file2.{lang_ext}
```
"""

    obtain_irrelevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of folders that are irrelevant to fixing the problem.
Note that irrelevant folders are those that do not need to be modified and are safe to ignored when trying to solve this problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path.
Remember that any subfolders will be considered as irrelevant if you provide the parent folder.
Please ensure that the provided irrelevant folders do not include any important files needed to fix the problem
The returned folders should be separated by new lines and wrapped with ```
For example:
```
folder1/
folder2/folder3/
folder4/folder5/
```
"""

    file_content_template = """
### File: {file_name} ###
{file_content}
"""
    file_content_in_block_template = """
### File: {file_name} ###
```{language}
{file_content}
```
"""

    obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
The possible location outputs should be either "class", "function" or "line".

### Examples:
```
full_path1/file1.{lang_ext}
line: 10
class: MyClass1
line: 51

full_path2/file2.{lang_ext}
function: MyClass2.my_method
line: 12

full_path3/file3.{lang_ext}
function: my_function
line: 24
line: 156
```

Return just the location(s) wrapped with ```.
"""

    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class, method, or function names that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class, method, or function names that need to be edited.
### Examples:
```
full_path1/file1.{lang_ext}
function: my_function1
class: MyClass1

full_path2/file2.{lang_ext}
function: MyClass2.my_method
class: MyClass3

full_path3/file3.{lang_ext}
function: my_function2
```

Return just the location(s) wrapped with ```.
"""
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.{lang_ext}
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.{lang_ext}
variable: my_var
function: MyClass3.my_method

full_path3/file3.{lang_ext}
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

    obtain_relevant_functions_and_vars_from_raw_files_prompt = """
Please look through the following GitHub Problem Description and Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.{lang_ext}
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.{lang_ext}
variable: my_var
function: MyClass3.my_method

full_path3/file3.{lang_ext}
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations wrapped with ```.
"""

    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 300
        self.model_name = model_name
        self.backend = backend
        self.logger = logger

    def _parse_model_return_lines(self, content: str) -> list[str]:
        """解析 LLM 返回的文件路径列表，支持 ``` 代码块格式"""
        if not content:
            return None
        
        content = content.strip()
        
        # 尝试提取 ``` 代码块中的内容
        import re
        code_block_pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        if matches:
            # 使用代码块中的内容
            content = matches[0].strip()
        else:
            # 如果没有代码块，尝试查找第一个 ``` 之后的内容
            if '```' in content:
                parts = content.split('```')
                if len(parts) >= 2:
                    content = parts[1].strip()
                    # 移除可能的语言标识符（第一行）
                    lines = content.split('\n')
                    if lines and not any(c.isalnum() or c in './' for c in lines[0].strip()):
                        content = '\n'.join(lines[1:]).strip()
        
        # 按行分割并过滤空行和标记
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        # 移除可能的 ``` 标记
        lines = [line for line in lines if not line.startswith('```')]
        
        return lines if lines else None

    def localize_irrelevant(self, top_n=1, mock=False):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        message = self.obtain_irrelevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=2048,  # self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        f_files = []
        filtered_files = []

        model_identified_files_folder = self._parse_model_return_lines(raw_output)
        self.logger.info(f"Parsed model identified folders/files: {model_identified_files_folder}")
        
        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )
        self.logger.info(f"Total files in structure: {len(files)}")
        if len(files) == 0:
            self.logger.warning(f"WARNING: Repository structure is empty for {self.instance_id}!")
            # 如果结构为空，返回空列表
            return (
                [],
                {
                    "raw_output_files": raw_output,
                    "found_files": [],
                    "filtered_files": [],
                },
                traj,
            )
        else:
            self.logger.info(f"First 5 files in structure: {[f[0] for f in files[:5]]}")
        
        # 如果模型没有返回任何内容，所有文件都是 relevant
        if model_identified_files_folder is None or len(model_identified_files_folder) == 0:
            self.logger.warning(f"WARNING: Model returned no irrelevant folders/files for {self.instance_id}, treating all files as relevant")
            f_files = [f[0] for f in files]
            self.logger.info(f"All {len(f_files)} files treated as relevant")
        else:
            # 处理模型返回的内容：可能是文件夹（以/结尾）或文件路径
            # 判断：如果主要是文件路径，可能是 relevant files（误返回），应该反转逻辑
            processed_folders = []
            valid_paths = False  # 标记是否有有效的路径
            file_path_count = 0  # 文件路径数量
            folder_path_count = 0  # 文件夹路径数量
            
            for x in model_identified_files_folder:
                x = x.strip()
                if not x:
                    continue
                
                # 检查是否是有效的路径（包含 / 或 . 或字母数字）
                # 排除明显的错误消息（如包含 "missing", "error", "cannot" 等）
                error_keywords = ['missing', 'error', 'cannot', 'unable', 'please', 'tell me', 'where']
                if any(keyword in x.lower() for keyword in error_keywords):
                    self.logger.warning(f"WARNING: Model response appears to be an error message, not a path: {x[:100]}")
                    continue
                
                # 检查是否看起来像路径（包含 / 或 . 或至少包含字母数字和常见路径字符）
                if '/' in x or '.' in x or any(c.isalnum() for c in x):
                    valid_paths = True
                    # 判断是文件路径还是文件夹路径
                    if end_with_ext(x) and not x.endswith("/"):
                        # 文件路径
                        file_path_count += 1
                        # 取目录部分作为 irrelevant folder
                        import os
                        folder = os.path.dirname(x)
                        if folder:
                            folder = folder + "/" if not folder.endswith("/") else folder
                            processed_folders.append(folder)
                        # 也添加文件本身作为 irrelevant（精确匹配）
                        processed_folders.append(x)
                    elif x.endswith("/"):
                        # 文件夹路径
                        folder_path_count += 1
                        processed_folders.append(x)
                    else:
                        # 可能是文件夹但没有/结尾
                        folder_path_count += 1
                        processed_folders.append(x + "/")
            
            # 如果没有有效的路径，将所有文件视为 relevant
            if not valid_paths or len(processed_folders) == 0:
                self.logger.warning(f"WARNING: No valid paths found in model response for {self.instance_id}, treating all files as relevant")
                self.logger.warning(f"Model response: {raw_output[:200]}")
                f_files = [f[0] for f in files]
                self.logger.info(f"All {len(f_files)} files treated as relevant")
            else:
                # 初始化 relevant_file_paths 为空列表
                relevant_file_paths = []
                
                # 判断：如果主要是文件路径（>50%），可能是误返回的 relevant files
                # 在这种情况下，应该将这些文件视为 relevant，而不是 irrelevant
                total_paths = file_path_count + folder_path_count
                if total_paths > 0 and file_path_count > folder_path_count and file_path_count / total_paths > 0.5:
                    self.logger.warning(f"WARNING: Model returned mostly file paths ({file_path_count}/{total_paths}), treating them as relevant files instead of irrelevant")
                    # 提取文件路径作为 relevant files
                    for x in model_identified_files_folder:
                        x = x.strip()
                        if end_with_ext(x) and not x.endswith("/"):
                            relevant_file_paths.append(x)

                # 匹配 repository 中的文件
                for file_content in files:
                    file_name = file_content[0]
                    if file_name in relevant_file_paths:
                        f_files.append(file_name)
                    else:
                        # 其他文件也视为 relevant（因为 LLM 只返回了部分 relevant files）
                        f_files.append(file_name)

                self.logger.info(f"Found {len(f_files)} relevant files based on model's file path response")

                # 正常情况：返回的是 irrelevant folders
                model_identified_files_folder = list(set(processed_folders))  # 去重
                self.logger.info(f"Processed model identified folders/files: {model_identified_files_folder}")

                for file_content in files:
                    file_name = file_content[0]
                    # 检查文件是否匹配任何 irrelevant 路径
                    is_irrelevant = False
                    for irrelevant_path in model_identified_files_folder:
                        if irrelevant_path.endswith("/"):
                            # 文件夹路径：检查文件是否在这个文件夹下
                            if file_name.startswith(irrelevant_path):
                                is_irrelevant = True
                                break
                        else:
                            # 文件路径：精确匹配
                            if file_name == irrelevant_path:
                                is_irrelevant = True
                                break
                        
                        if is_irrelevant:
                            filtered_files.append(file_name)
                        else:
                            f_files.append(file_name)
                    
                    self.logger.info(f"Filtered files (irrelevant): {len(filtered_files)}, Remaining files (relevant): {len(f_files)}")
                    if len(f_files) == 0 and len(filtered_files) > 0:
                        self.logger.warning(f"WARNING: All files were filtered out as irrelevant for {self.instance_id}!")
                        # 如果所有文件都被过滤掉，至少保留一些文件作为 relevant
                        self.logger.warning(f"FALLBACK: Treating all files as relevant to avoid empty result")
                        f_files = [f[0] for f in files]
                        filtered_files = []
                    elif len(f_files) == 0 and len(model_identified_files_folder) > 0:
                        self.logger.warning(f"WARNING: Model identified folders but no files matched!")
                        self.logger.warning(f"Model folders: {model_identified_files_folder[:10]}")
                        self.logger.warning(f"Structure files (first 10): {[f[0] for f in files[:10]]}")
                        # 如果匹配失败，将所有文件视为 relevant
                        self.logger.warning(f"FALLBACK: Treating all files as relevant")
                        f_files = [f[0] for f in files]
                        filtered_files = []

            self.logger.info(raw_output)

        return (
            f_files,
            {
                "raw_output_files": raw_output,
                "found_files": f_files,
                "filtered_files": filtered_files,
            },
            traj,
        )

    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        found_files = []

        message = self.obtain_relevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
            lang_ext=LANG_EXT[0],
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        
        # DEBUG: 打印原始输出
        self.logger.info(f"==== RAW OUTPUT ====")
        self.logger.info(f"Content length: {len(raw_output) if raw_output else 0}")
        self.logger.info(f"Content repr: {repr(raw_output[:500]) if raw_output else 'None'}")
        self.logger.info(f"Content: {raw_output}")
        self.logger.info(f"==== END RAW OUTPUT ====")
        
        model_found_files = self._parse_model_return_lines(raw_output)
        self.logger.info(f"Parsed model found files: {model_found_files}")

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )
        self.logger.info(f"Total files in structure: {len(files)}")
        if len(files) == 0:
            self.logger.warning(f"WARNING: Repository structure is empty for {self.instance_id}!")
            # 如果 repository structure 为空，但 LLM 返回了文件路径，直接使用这些路径
            if model_found_files and len(model_found_files) > 0:
                self.logger.warning(f"FALLBACK: Using model's file paths directly since repository structure is empty")
                found_files = model_found_files
            else:
                found_files = []
        else:
            self.logger.info(f"First 5 files in structure: {[f[0] for f in files[:5]]}")

        # sort based on order of appearance in model_found_files
        found_files = correct_file_paths(model_found_files, files)
        self.logger.info(f"Matched found_files: {found_files}")
        if model_found_files and len(found_files) == 0:
            self.logger.warning(f"WARNING: Model returned {len(model_found_files)} files but none matched!")
            self.logger.warning(f"Model files: {model_found_files[:10]}")
            self.logger.warning(f"Structure files (first 10): {[f[0] for f in files[:10]]}")
            # 修复：不要 fallback 到不存在的路径，因为这会导致后续阶段失败
            # 如果匹配失败，返回空列表，让后续处理逻辑处理这种情况
            self.logger.warning(f"SKIP FALLBACK: Not using model's file paths since they don't exist in the repo")
            found_files = []

        self.logger.info(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_function_from_compressed_files(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
        compress_assign: bool = False,
        total_lines=30,
        prefix_lines=10,
        suffix_lines=10,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(
                code,
                compress_assign=compress_assign,
                total_lines=total_lines,
                prefix_lines=prefix_lines,
                suffix_lines=suffix_lines,
            )
            for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code, language=LANGUAGE)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents, lang_ext=LANG_EXT[0]
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents, lang_ext=LANG_EXT[0]
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names, keep_old_order
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_function_from_raw_text(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        raw_file_contents = {fn: code for fn, code in file_contents.items()}
        contents = [
            self.file_content_template.format(file_name=fn, file_content=code)
            for fn, code in raw_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = self.obtain_relevant_functions_and_vars_from_raw_files_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents, lang_ext=LANG_EXT[0]
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents, lang_ext=LANG_EXT[0]
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names, keep_old_order
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_line_from_coarse_function_locs(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content, lang_ext=LANG_EXT[0]
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(coarse_locs) > 1:
            self.logger.info(f"reducing to \n{len(coarse_locs)} files")
            coarse_locs.popitem()
            topn_content, file_loc_intervals = construct_topn_file_context(
                coarse_locs,
                file_names,
                file_contents,
                self.structure,
                context_window=context_window,
                loc_interval=True,
                add_space=add_space,
                sticky_scroll=sticky_scroll,
                no_line_number=no_line_number,
            )
            message = template.format(
                problem_statement=self.problem_statement, file_contents=topn_content, lang_ext=LANG_EXT[0]
            )

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(
            message, num_samples=num_samples, prompt_cache=num_samples > 1
        )

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names, keep_old_order
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )

    def localize_line_from_raw_text(
        self,
        file_names,
        mock=False,
        temperature=0.0,
        num_samples=1,
        keep_old_order=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        raw_file_contents = {
            fn: line_wrap_content(code) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_template.format(file_name=fn, file_content=code)
            for fn, code in raw_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents, lang_ext=LANG_EXT[0]
        )
        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents, lang_ext=LANG_EXT[0]
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names, keep_old_order
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)

        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
