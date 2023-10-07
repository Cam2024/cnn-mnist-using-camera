from PIL import Image
import numpy as np

# 1. 读取图片
image = Image.open('R2.png')

# 2. 将图片调整为28x28像素的大小
image = image.resize((28, 28))

# 3. 灰度化
image = image.convert('L')

# 4. 阈值化（可选）

# 5. 归一化
image = np.array(image) / 255.0

# 6. 保存图像
image = (image * 255).astype(np.uint8)
output_image = Image.fromarray(image)
output_image.save('mnist3.png')
