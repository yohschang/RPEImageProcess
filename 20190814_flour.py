from btimage import PhaseRetrieval, TimeLapseCombo, BT_image, CellLabelOneImage
from matplotlib import pyplot as plt
import cv2


root = r"E:\DPM\20190814_pointgray_test"

id_im = BT_image(root + "\\id.png")
id_im.open_image(color='g')

bead_im = BT_image(root + "\\power11_exposure1s_bead.png")
bead_im.open_image(color='g')
#
# bead = bead_im.img - id_im.img
# cv2.imwrite(r"E:\DPM\20190814_pointgray_test\bead_withoutid.png", bead)

foo1 = BT_image(root + "\\power100_exposure1s_bead_ND2_1.png")
foo1.open_image(color='g')
foo2 = BT_image(root + "\\power100_exposure1s_bead_ND2_2.png")
foo2.open_image(color='g')
foo3 = BT_image(root + "\\power100_exposure1s_bead_ND2_3.png")
foo3.open_image(color='g')
foo4 = BT_image(root + "\\power100_exposure1s_bead_ND2_4.png")
foo4.open_image(color='g')
foo5 = BT_image(root + "\\power100_exposure1s_bead_ND2_5.png")
foo5.open_image(color='g')

# f1 = foo1.img + foo2.img + foo3.img + foo4.img + foo5.img
f1 = foo1.img
f1 = cv2.medianBlur(f1, 11)

fig, axe = plt.subplots(1, 3, figsize=(12, 4))
axe[0].set_title("original bead")
axe[0].imshow(bead_im.img, cmap="gray")
axe[0].axis("off")
axe[1].set_title("ND2 attenuation")
axe[1].imshow(foo1.img, cmap="gray")
axe[1].axis("off")
axe[2].set_title("ND2 attenuation after processing")
axe[2].imshow(f1, cmap="gray")
axe[2].axis("off")
fig.tight_layout()
plt.show()



