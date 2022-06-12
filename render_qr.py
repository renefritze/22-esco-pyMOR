import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.colormasks import RadialGradiantColorMask


def make_link(url:str, filename:str)->None:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,)
    qr.add_data(url)
    img_2 = qr.make_image(#image_factory=StyledPilImage, #color_mask=RadialGradiantColorMask(),
        back_color=(255, 255, 255), fill_color=(0, 0, 0))
    img_2.save(filename)

make_link('https://school.pymor.org', 'qr_school_website.png')
make_link('https://docs.pymor.org', 'qr_docs.png')