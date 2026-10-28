import json
import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "real_estate_data.csv"))
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"
app = FastAPI(title="Прогноз стоимости недвижимости")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metadata = {}
if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
data = pd.DataFrame()
if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH, low_memory=False)
def normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()
def sale_mask(df: pd.DataFrame) -> pd.Series:
    if "listing_type" not in df.columns:
        return pd.Series(False, index=df.index)
    return normalized(df["listing_type"]).eq("1")
def currency_mask(df: pd.DataFrame, currency="TRY") -> pd.Series:
    if "price_currency" not in df.columns:
        return pd.Series(False, index=df.index)
    return normalized(df["price_currency"]).str.upper().eq(currency.upper())
def get_choices():
    if data.empty:
        return {"sub_types": [], "cities": [], "heating": []}
    sale = data.loc[sale_mask(data)].copy()
    if sale.empty:
        sale = data.copy()
    sub_types = sorted(normalized(sale["sub_type"]).dropna().replace("", pd.NA).dropna().unique().tolist()) if "sub_type" in sale else []
    cities = []
    if "address" in sale:
        cities = sorted(normalized(sale["address"]).dropna().replace("", pd.NA).dropna().str.split("/", n=1).str[0].dropna().unique().tolist())
    heating = sorted(normalized(sale["heating_type"]).dropna().replace("", pd.NA).dropna().unique().tolist()) if "heating_type" in sale else []
    return {"sub_types": sub_types, "cities": cities, "heating": heating}
def room_to_number(room_count: str) -> float:
    value = str(room_count).strip()
    if "+" not in value:
        raise ValueError("Укажите количество комнат в формате, например 2+1.")
    first = value.split("+", 1)[0].strip().replace(",", ".")
    number = float(first)
    if number <= 0 or number > 20:
        raise ValueError("Количество комнат должно быть от 1 до 20.")
    return number
def make_raw_model_row(size, room_count, building_age, sub_type, city):
    return pd.DataFrame([{
        "size": float(size),
        "room_num": room_to_number(room_count),
        "building_age": float(building_age),
        "sub_type": str(sub_type).strip(),
        "city": str(city).strip(),
    }])
def dashboard_data():
    if data.empty:
        return {"rows": 0, "sale_try": 0, "mean_price": 0.0, "median_price": 0.0}, {"city": {"x": [], "y": []}, "subtype": {"x": [], "y": []}}
    mask = sale_mask(data) & currency_mask(data)
    sale = data.loc[mask].copy()
    sale["price"] = pd.to_numeric(sale.get("price"), errors="coerce")
    sale = sale.dropna(subset=["price"])
    stats = {
        "rows": int(len(data)),
        "sale_try": int(len(sale)),
        "mean_price": float(sale["price"].mean()) if not sale.empty else 0.0,
        "median_price": float(sale["price"].median()) if not sale.empty else 0.0,
    }
    if "address" in sale:
        sale["city"] = normalized(sale["address"]).str.split("/", n=1).str[0]
        city = sale.dropna(subset=["city"]).groupby("city")["price"].mean().sort_values(ascending=False).head(15)
    else:
        city = pd.Series(dtype=float)
    if "sub_type" in sale:
        sale["sub_type"] = normalized(sale["sub_type"])
        subtype = sale.dropna(subset=["sub_type"]).groupby("sub_type")["price"].mean().sort_values(ascending=False).head(15)
    else:
        subtype = pd.Series(dtype=float)
    charts = {
        "city": {"x": city.index.tolist(), "y": [float(v) for v in city.tolist()]},
        "subtype": {"x": subtype.index.tolist(), "y": [float(v) for v in subtype.tolist()]},
    }
    return stats, charts
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "choices": get_choices(), "result": None, "error": None})
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    size: float = Form(...),
    room_count: str = Form(...),
    building_age: float = Form(...),
    sub_type: str = Form(...),
    city: str = Form(...),
    total_floor_count: str = Form(""),
    floor_no: str = Form(""),
    tom: float = Form(0),
    heating_type: str = Form(""),
    listing_type: str = Form("1"),
):
    error = None
    result = None
    try:
        if size <= 0 or size > 1000:
            raise ValueError("Площадь должна быть в диапазоне от 0 до 1000 м²")
        if building_age < 0 or building_age > 100:
            raise ValueError("Возраст здания должен быть от 0 до 100 лет.")
        if tom < 0:
            raise ValueError("Время на рынке не может быть отрицательным.")
        if not str(sub_type).strip():
            raise ValueError("Выберите тип жилья.")
        if not str(city).strip():
            raise ValueError("Выберите город.")
        X = make_raw_model_row(size, room_count, building_age, sub_type, city)
        prediction = float(model.predict(X)[0])
        mae = float(metadata.get("metrics", {}).get("MAE", 0))
        result = {
            "price": max(0.0, prediction),
            "low": max(0.0, prediction - mae),
            "high": max(0.0, prediction + mae),
            "mae": mae,
            "city": city,
            "heating": heating_type or "Не указано",
        }
    except (ValueError, TypeError) as exc:
        error = str(exc)
    return templates.TemplateResponse("index.html", {"request": request, "choices": get_choices(), "result": result, "error": error})
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    stats, charts = dashboard_data()
    metrics = metadata.get("metrics", {}) or {}
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats, "charts": charts, "metrics": metrics, "metadata": metadata})
@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return templates.TemplateResponse("help.html", {"request": request, "metadata": metadata})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
