import torch

class TextInputNode:
    def __init__(self):
        self.type = "TextInput"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入文本，每行一个项目"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "output_text"
    CATEGORY = "text"

    def output_text(self, text):
        return (text,)

class TextSortNode:
    def __init__(self):
        self.type = "TextSort"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_count": ("INT", {
                    "default": 3,
                    "min": 3,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "label": "文本框数量"
                }),
                **{f"text{i+1}": ("STRING", {
                    "multiline": True,
                    "default": "",
                }) for i in range(10)},
                "sort_mode": ([
                    "按字母升序",
                    "按字母降序",
                    "按长度升序",
                    "按长度降序",
                    "按输入顺序"
                ], {"default": "按字母升序"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "sort_text"
    CATEGORY = "text"

    def sort_text(self, sort_mode, **kwargs):
        # 获取所有文本输入
        text_count = kwargs.get('text_count', 3)
        texts = [kwargs.get(f'text{i+1}', '') for i in range(text_count)]
        
        # 合并所有输入的文本
        all_text = '\n'.join(filter(None, texts))
        
        # 将文本分割成行
        lines = all_text.split('\n')
        lines = [line for line in lines if line.strip()]
        
        # 根据选择的模式进行排序
        if sort_mode == "按字母升序":
            lines.sort()
        elif sort_mode == "按字母降序":
            lines.sort(reverse=True)
        elif sort_mode == "按长度升序":
            lines.sort(key=len)
        elif sort_mode == "按长度降序":
            lines.sort(key=len, reverse=True)
        elif sort_mode == "按输入顺序":
            pass  # 保持原始输入顺序，不需要额外排序
        
        # 将排序后的行重新组合成文本
        sorted_text = '\n'.join(lines)
        return (sorted_text,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "TextInput": TextInputNode,
    "TextSort": TextSortNode
}

# 设置显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextInput": "文本输入",
    "TextSort": "文本排序"
}