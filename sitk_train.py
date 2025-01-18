import SimpleITK as sitk
import numpy as np


# 图像重采样
def resample_image(itk_image, out_spacing=None):
    if out_spacing is None:
        out_spacing = [1.0, 1.0, 2.0]
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)


gz_path = 'C:\\Users\\10100\\Downloads\\a.jpg'
print('测试文件名为：', gz_path)
Original_img = sitk.ReadImage(gz_path)
print('原始图像的Spacing：', Original_img.GetSpacing())
print('原始图像的Size：', Original_img.GetSize())
Resample_img = resample_image(Original_img)
print('经过重采样之后图像的Spacing是：', Resample_img.GetSpacing())
print('经过重采样之后图像的Size是：', Resample_img.GetSize())
