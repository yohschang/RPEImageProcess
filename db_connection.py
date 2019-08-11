from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from databaseORM import RetinalPigmentEpithelium
import pandas as pd

engine = create_engine('mysql+pymysql://BT:x1x4x5x6@127.0.0.1:3306/Cell')
Session = sessionmaker(bind=engine, autoflush=False)
sess = Session()

image_path = "E:\\DPM\\20190708_time_lapse_succ\\Bead\\1\\SP\\time-lapse\\phase_npy\\1_phase.npy"
label_path = "E:\\DPM\\20190708_time_lapse_succ\\Bead\\1\\SP\\time-lapse\\afterwater\\1_afterwater.npy"

# cell_test = RetinalPigmentEpithelium(0, 2019, 8, 10, 0, 0, image_path, label_path)
# sess.add(cell_test)
a = sess.query(RetinalPigmentEpithelium).all()

sess.commit()
engine.dispose()



