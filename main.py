from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms

def get_high_pass_image(image):
    gray_image = image.convert('L')
    image_array = np.array(gray_image)
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)
    rows, cols = image_array.shape
    crow, ccol = int(rows/2), int(cols/2)
    high_pass_filter = np.ones_like(image_array)
    high_pass_filter[crow-100:crow+100, ccol-100:ccol+100] = 0
    f_transform_filtered = f_transform_shifted * high_pass_filter
    f_transform = np.fft.ifftshift(f_transform_filtered)
    image_f_transform = np.fft.ifft2(f_transform)
    image_f_transform = np.abs(image_f_transform)
    return Image.fromarray(np.uint8(image_f_transform))

class HighPassNet(nn.Module):
    def __init__(self):
        super(HighPassNet, self).__init__()
        self.conv2d = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        return self.conv2d(x)
    
if __name__ == "__main__":
    image = Image.open("data/navia.png")
    high_pass_image = get_high_pass_image(image)

    # 可视化数据
    titles = ["Original Image", "High Pass Image"]
    fig, axes = plt.subplots(2, 2, figsize=(30, 10))
    for ax, img, title in zip(axes[0], [image, high_pass_image], titles):
        ax.imshow(img)
        ax.set_title(f'{title}\nShape: {img.size}\nChannels: {len(img.getbands())}')
        ax.axis("off")
    # plt.show()

    trans = transforms.Compose([transforms.ToTensor()])

    net = HighPassNet()

    image_tensor = trans(image).unsqueeze(0)
    high_pass_image_tensor = trans(high_pass_image).unsqueeze(0)
    high_pass_image_tensor = high_pass_image_tensor.expand(-1, 3, -1, -1)

    print(image_tensor.shape, high_pass_image_tensor.shape)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        output = net(image_tensor)
        loss = criterion(output, high_pass_image_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    image = Image.open("data/clorinde.jpg")
    net.eval()
    with torch.no_grad():
        output = net(trans(image).unsqueeze(0))
    output_image = output.squeeze(0).detach().numpy().transpose(1, 2, 0)
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    # output_image.show()

    for ax, img, title in zip(axes[1], [image, output_image], titles):
        ax.imshow(img)
        ax.set_title(f'{title}\nShape: {img.size}\nChannels: {len(img.getbands())}')
        ax.axis("off")

    plt.show()
    

