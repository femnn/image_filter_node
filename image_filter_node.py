import numpy as np
import torch

class ImageColorFilter:
    """
    一个用于调整图片颜色的ComfyUI节点
    可以调整图片的色相、饱和度和亮度
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "label": "色相调整",
                    "color": "#FF6B6B"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "label": "饱和度",
                    "color": "#4ECDC4"
                }),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "label": "亮度",
                    "color": "#FFD93D"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"
    CATEGORY = "image/adjustments"

    def rgb_to_hsv(self, rgb):
        # 确保输入是PyTorch张量
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        
        # 获取维度信息
        original_shape = rgb.shape
        if len(rgb.shape) == 3:
            rgb = rgb.reshape(-1, 3)
        elif len(rgb.shape) == 4:
            rgb = rgb.reshape(-1, 3)
        
        # 分离RGB通道
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        # 计算HSV值
        maxc, _ = torch.max(rgb, dim=1)
        minc, _ = torch.min(rgb, dim=1)
        v = maxc
        deltac = maxc - minc
        
        # 计算饱和度
        s = torch.where(maxc != 0, deltac / maxc, torch.zeros_like(maxc))
        deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
        
        # 计算色相
        h = torch.zeros_like(maxc)
        
        # r是最大值的情况
        mask_r = (maxc == r)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / deltac[mask_r]) % 6
        
        # g是最大值的情况
        mask_g = (maxc == g)
        h[mask_g] = ((b[mask_g] - r[mask_g]) / deltac[mask_g]) + 2
        
        # b是最大值的情况
        mask_b = (maxc == b)
        h[mask_b] = ((r[mask_b] - g[mask_b]) / deltac[mask_b]) + 4
        
        h = h / 6.0
        
        # 组合HSV通道
        hsv = torch.stack([h, s, v], dim=1)
        
        # 恢复原始维度
        hsv = hsv.reshape(original_shape)
        
        return hsv

    def hsv_to_rgb(self, hsv):
        # 确保输入是PyTorch张量
        if isinstance(hsv, np.ndarray):
            hsv = torch.from_numpy(hsv)
        
        # 获取维度信息
        original_shape = hsv.shape
        if len(hsv.shape) == 3:
            hsv = hsv.reshape(-1, 3)
        elif len(hsv.shape) == 4:
            hsv = hsv.reshape(-1, 3)
        
        # 分离HSV通道
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        h = h * 6.0
        i = torch.floor(h)
        f = h - i
        
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        
        # 初始化RGB通道
        rgb = torch.zeros_like(hsv)
        
        # 根据色相区间设置RGB值
        mask = (i == 0)
        rgb[mask] = torch.stack([v[mask], t[mask], p[mask]], dim=1)
        
        mask = (i == 1)
        rgb[mask] = torch.stack([q[mask], v[mask], p[mask]], dim=1)
        
        mask = (i == 2)
        rgb[mask] = torch.stack([p[mask], v[mask], t[mask]], dim=1)
        
        mask = (i == 3)
        rgb[mask] = torch.stack([p[mask], q[mask], v[mask]], dim=1)
        
        mask = (i == 4)
        rgb[mask] = torch.stack([t[mask], p[mask], v[mask]], dim=1)
        
        mask = (i == 5)
        rgb[mask] = torch.stack([v[mask], p[mask], q[mask]], dim=1)
        
        # 恢复原始维度
        rgb = rgb.reshape(original_shape)
        
        return rgb

    def apply_filter(self, image, hue, saturation, brightness):
        # 确保输入是PyTorch张量
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # 转换为HSV颜色空间
        hsv = self.rgb_to_hsv(image)
        
        # 应用颜色调整
        hsv[..., 0] = (hsv[..., 0] + hue) % 1.0  # 调整色相
        hsv[..., 1] = torch.clamp(hsv[..., 1] * saturation, 0, 1)  # 调整饱和度
        hsv[..., 2] = torch.clamp(hsv[..., 2] * brightness, 0, 1)  # 调整亮度
        
        # 转换回RGB颜色空间
        result = self.hsv_to_rgb(hsv)
        
        # 确保值在有效范围内
        result = torch.clamp(result, 0, 1)
        
        # 如果需要，转换回NumPy数组
        if isinstance(image, np.ndarray):
            result = result.numpy()
        
        return (result,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageColorFilter": ImageColorFilter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageColorFilter": "图片颜色滤镜"
}