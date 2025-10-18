from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import APP_CONFIG, CORS_ORIGINS
from core.startup import load_subapps
from api.routes.llm_routes import coordinator_route

app = FastAPI(**APP_CONFIG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar routers
app.include_router(coordinator_route.router)


# Montar subapps dinámicas
availability = load_subapps(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
