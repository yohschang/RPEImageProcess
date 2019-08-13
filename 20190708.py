import os
import cv2
import numpy as np
from btimage import check_file_exist
from btimage import BT_image, CellLabelOneImage, PrevNowCombo, TimeLapseCombo, Fov, WorkFlow, AnalysisCellFeature
from btimage import TimeLapseCombo
import glob
from matplotlib import pyplot as plt
from os import getenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from databaseORM import RetinalPigmentEpithelium

def normalize(array):
    max_value = max(array)
    min_value = min(array)
    return list(map(lambda old: (old - min_value) / (max_value - min_value), array))


root_path = "E:\\DPM\\20190708_time_lapse_succ\\Bead\\1\\SP\\time-lapse\\"
####################################################################################

# t = TimeLapseCombo(root_path=root_path)
# t.read(1, 36)
# t.combo(target=25, save=True, strategy="cheat")


# img28 = np.load(r"E:\DPM\20190708_time_lapse_succ\Bead\1\SP\time-lapse\phase_npy\28_phase.npy")
# plt.figure()
# plt.imshow(img28, cmap='jet', vmax=3.0, vmin=-0.5)
# plt.show()
# img27 = np.load(r"E:\DPM\20190708_time_lapse_succ\Bead\1\SP\time-lapse\phase_npy\27_phase.npy")
# plt.figure()
# plt.imshow(img27, cmap='jet', vmax=3.0, vmin=-0.5)
# plt.show()
# img26 = np.load(r"E:\DPM\20190708_time_lapse_succ\Bead\1\SP\time-lapse\phase_npy\26_phase.npy")
# plt.figure()
# plt.imshow(img26, cmap='jet', vmax=3.0, vmin=-0.5)
# plt.show()
####################################################################################
# f = Fov(root_path, 1, 36)
# f.run()


###################################################################################
# label and match

# current_target = 30

# after = CellLabelOneImage(root_path, target=current_target).run(adjust=True, plot_mode=False, load="old", save_water=True)
# output = PrevNowCombo(root_path).combo(now_target=current_target, save=True)

# plt.close()
# plt.figure()
# plt.title(str(current_target) + "label img")
# plt.imshow(after, cmap='jet')
# plt.colorbar()
# plt.show()
# ####################################################################################
# analysis

# ana = AnalysisCellFeature(root_path)


####################################################################################
# acf = AnalysisCellFeature(root_path)
# acf.image_by_image(db_save=True, png_save=True, plot_mode=False)
# acf.check_last_id()

####################################################################################
# peek the data
passward = getenv("DBPASS")
engine = create_engine('mysql+pymysql://BT:' + passward + '@127.0.0.1:3306/Cell')
Session = sessionmaker(bind=engine, autoflush=False)
sess = Session()
#
# a = sess.query(RetinalPigmentEpithelium).all()
# print(len(a))

a = sess.query(RetinalPigmentEpithelium).filter(RetinalPigmentEpithelium.year == 2019
                                                , RetinalPigmentEpithelium.month == 7
                                                , RetinalPigmentEpithelium.day == 8
                                                , RetinalPigmentEpithelium.label == 64).order_by(RetinalPigmentEpithelium.id.asc()).all()
mean_optical_height = []
phase_std = []
cir = []
distance_coef = []
area = []

for cell in a:
    print(cell.id)
    mean_optical_height.append(cell.mean_optical_height)
    phase_std.append(cell.phase_std)
    cir.append(cell.circularity)
    distance_coef.append(cell.distance_coef)
    area.append(cell.area)
    print(cell.img_path)


mean_optical_height = normalize(mean_optical_height)
phase_std = normalize(phase_std)
cir = normalize(cir)
distance_coef = normalize(distance_coef)
area = normalize(area)

plt.figure()
plt.plot(mean_optical_height, label="optical height")
plt.plot(phase_std, label="phase std")
plt.plot(cir, label="cir")
# plt.plot(distance_coef, label="distance_coef")
plt.plot(area, label="area")
plt.legend()
plt.show()


## Cannot plot all the cell because of their different apoptosis time.
# plt.figure()
# for label in range(2, 90):
#     a = sess.query(RetinalPigmentEpithelium).filter(RetinalPigmentEpithelium.year == 2019
#                                                     , RetinalPigmentEpithelium.month == 7
#                                                     , RetinalPigmentEpithelium.day == 8
#                                                     , RetinalPigmentEpithelium.label == label).order_by(RetinalPigmentEpithelium.id.asc()).all()
#
#     if len(a) == 30:
#         phase_mean = []
#         for cell in a:
#             phase_mean.append(cell.phase_mean)
#         phase_mean = normalize(phase_mean)
#         plt.plot(phase_mean)
#
# plt.show()

# engine.dispose()

