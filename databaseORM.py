from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, Integer, Boolean


engine = create_engine('mysql+pymysql://BT:x1x4x5x6@127.0.0.1:3306/Cell')
# Session = sessionmaker(bind=engine, autoflush=False)
# sess = Session()
Base = declarative_base()


class RetinalPigmentEpithelium(Base):
    __tablename__ = 'RetinalPigmentEpithelium'
    id = Column('ID', Integer, primary_key=True, index=True)
    year = Column('Year', Integer)
    month = Column('Month', Integer)
    day = Column('Day', Integer)
    label = Column('Label', Integer)
    time_lapse_num = Column('Time_lapse_num', Integer)

    phase_mean = Column('phase_mean', Float)
    phase_std = Column('phase_std', Float)
    circularity = Column('circularity', Float)
    area = Column('area', Float)
    apoptosis = Column("apoptosis", Boolean)
    mean_optical_height = Column("mean_optical_height", Float)
    distance_coef = Column("distance_coef", Float)

    img_path = Column('img_path', String(150))
    label_path = Column("label_path", String(150))

    def __init__(self, id, year, month, day, label, time_lapse_num, img_path, label_path, features):
        self.id = id
        self.year = year
        self.month = month
        self.day = day
        self.label = label
        self.time_lapse_num = time_lapse_num
        self.img_path = img_path
        self.label_path = label_path

        # features
        self.phase_mean = features[0]
        self.phase_std = features[1]
        self.circularity = features[2]
        self.area = features[3]
        self.apoptosis = features[4]
        self.mean_optical_height = features[5]
        self.distance_coef = features[6]

    def __repr__(self):
        repr_str1 = "<RetinalPigmentEpithelium("
        repr_str2 = "id: {}\n"
        repr_str3 = "date: {}\n"
        repr_str4 = "label: {}\n"
        repr_str5 = "time_lapse_num: {}\n"
        repr_str6 = "img_path: {}\n"
        repr_str7 = "phase_mean: {}\n"
        repr_str8 = "phase_std: {}\n"
        repr_str9 = "circularity: {}\n"
        repr_str10 = "area: {}\n"
        repr_str11 = ")>"
        repr_str = repr_str1 + repr_str2 + repr_str3 + repr_str4 + repr_str5 + repr_str6 + repr_str7 + repr_str8 + repr_str9 + repr_str10 + repr_str11
        return repr_str.format(self.id, str(self.year)+str(self.month)+str(self.day), self.label, self.time_lapse_num, self.img_path,
                               self.phase_mean, self.phase_std, self.circularity, self.area)


# connect engine to database obj
Base.metadata.create_all(engine)
# sess.commit()
engine.dispose()
