# Real-SRFlow
RectifiedFlow based Real Super-Resolution


>> - https://github.com/xinntao/Real-ESRGAN
>> - https://github.com/EternalEvan/FlowIE


INTER_NEAREST (0): 最近邻插值。速度最快，但可能会产生块状效果。
INTER_LINEAR (1): 双线性插值。速度和质量之间的平衡，常用于放大图像。
INTER_CUBIC (2): 双三次插值。基于4x4像素邻域，生成比双线性插值更平滑的图像，但速度较慢。
INTER_AREA (3): 基于像素区域关系的重采样。适用于图像缩小，能产生无摩尔纹的结果。
INTER_LANCZOS4 (4): Lanczos插值，基于8x8像素邻域。提供最高质量，但速度最慢。
INTER_LINEAR_EXACT (5): 类似于双线性插值，但计算更精确。
INTER_NEAREST_EXACT (6): 类似于最近邻插值，但计算更精确。