
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input data for a single order prediction.

    The fields below correspond to the features expected by
    the saved preprocessing pipeline.
    """

    total_items: float = Field(..., ge=0)
    total_price: float = Field(..., ge=0)
    total_freight_value: float = Field(..., ge=0)
    avg_item_price: float = Field(..., ge=0)

    total_payment_value: float = Field(..., ge=0)
    max_installments: float = Field(..., ge=0)
    payment_count: float = Field(..., ge=0)

    unique_products: float = Field(..., ge=0)
    unique_sellers: float = Field(..., ge=0)

    customer_city: str
    customer_state: str
    main_payment_type: str

    purchase_year: Optional[float] = None
    purchase_month: Optional[float] = None
    purchase_day: Optional[float] = None
    purchase_weekday: Optional[float] = None
    purchase_hour: Optional[float] = None
    is_weekend: Optional[float] = None

    multi_seller: Optional[float] = None
    freight_ratio: Optional[float] = None
    freight_per_item: Optional[float] = None

    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_late: float
    probability_on_time: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_version: str