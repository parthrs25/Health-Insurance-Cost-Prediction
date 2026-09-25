import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import yaml

logger = logging.getLogger(__name__)

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    age = Column(Integer)
    sex = Column(String)
    bmi = Column(Float)
    children = Column(Integer)
    smoker = Column(String)
    region = Column(String)
    predicted_cost = Column(Float)
    cost_low_bound = Column(Float)
    cost_high_bound = Column(Float)

def get_db_engine(db_path: str = "sqlite:///data/predictions.db"):
    os.makedirs("data", exist_ok=True)
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_dir = True
    Base.metadata.create_all(engine)
    return engine

def log_prediction(patient_data: dict, pred_cost: float, low_bound: float, high_bound: float, db_path: str = "sqlite:///data/predictions.db"):
    """Log prediction record to SQLite database."""
    engine = get_db_engine(db_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    record = PredictionRecord(
        age=patient_data["age"],
        sex=patient_data["sex"],
        bmi=patient_data["bmi"],
        children=patient_data["children"],
        smoker=patient_data["smoker"],
        region=patient_data["region"],
        predicted_cost=round(pred_cost, 2),
        cost_low_bound=round(low_bound, 2),
        cost_high_bound=round(high_bound, 2),
    )

    session.add(record)
    session.commit()
    session.close()
    logger.info(f"Logged prediction to DB: Cost=${pred_cost:.2f}")

def get_all_predictions(limit: int = 50, db_path: str = "sqlite:///data/predictions.db"):
    """Fetch history records from SQLite database."""
    engine = get_db_engine(db_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    records = session.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit).all()
    results = [
        {
            "id": r.id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
            "age": r.age,
            "sex": r.sex,
            "bmi": r.bmi,
            "children": r.children,
            "smoker": r.smoker,
            "region": r.region,
            "predicted_cost": r.predicted_cost,
            "cost_low_bound": r.cost_low_bound,
            "cost_high_bound": r.cost_high_bound,
        }
        for r in records
    ]
    session.close()
    return results

if __name__ == "__main__":
    engine = get_db_engine()
    print("SQLite Database initialized successfully at data/predictions.db")
