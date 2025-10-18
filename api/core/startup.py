def load_subapps(app):
    """Carga condicional de subaplicaciones y retorna el estado de disponibilidad"""
    availability = {}

    try:
        from api.routes.bitcoin_route.bitcoin_api import app as bitcoin_app
        app.mount("/bitcoin", bitcoin_app)
        availability["bitcoin"] = True
    except ImportError as e:
        print(f"Bitcoin API no disponible: {e}")
        availability["bitcoin"] = False
    try:
        from api.routes.movies_route.movies_api import app as movies_app
        app.mount("/movies", movies_app)
        availability["movies"] = True
    except ImportError as e:
        print(f"Movies API no disponible: {e}")
        availability["movies"] = False

    return availability