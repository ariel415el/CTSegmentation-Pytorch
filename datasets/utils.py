import SimpleITK as sitk


def read_volume(path):
    # volume = nib.load(path)
    # volume= volume.get_data()

    volume = sitk.ReadImage(path, sitk.sitkInt16)
    volume = sitk.GetArrayFromImage(volume)

    return volume