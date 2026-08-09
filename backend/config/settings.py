import sys
from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

# The agentic layer (`carmed`) sits at the repo root, next to backend/. In Docker
# it is bind-mounted into /app and already importable; this makes `manage.py`
# work when run from the host too. No-op when the directory isn't there.
if (REPO_ROOT / "carmed").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

env = environ.Env(
    DEBUG=(bool, True),
)
# Repo keeps a single .env at the root (shared with the /setup LLM key), not one per app.
environ.Env.read_env(REPO_ROOT / ".env")

SECRET_KEY = env("DJANGO_SECRET_KEY", default="dev-insecure-secret-key-change-me")
DEBUG = env.bool("DEBUG", default=True)
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["*"])

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.cases",
    "apps.diagnostics",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": env.db(
        "DATABASE_URL",
        default="postgres://carmaintenance:carmaintenance@localhost:5432/carmaintenance",
    ),
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Yerevan"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Django's default config only attaches a handler to the "django" logger, so
# anything our own apps log below WARNING is silently dropped. The agentic
# layer's trace is logged at INFO and is the main way to see what actually ran,
# so give apps.* a console handler of its own.
LOG_LEVEL = env("LOG_LEVEL", default="INFO")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(levelname)-7s %(name)s | %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "plain"},
    },
    "loggers": {
        "apps": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
    },
}

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
}

# --- Project-specific: embeddings for the Case KB / RAG (Agent 4 gate) ---
# Multilingual so the same index serves Armenian, Russian and English input (spec section 5).
EMBEDDING_MODEL_NAME = env(
    "EMBEDDING_MODEL_NAME", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM = env.int("EMBEDDING_DIM", default=384)

# Minimum cosine similarity a past diagnosis must reach to be treated as
# evidence about the current one. Applied once, at retrieval, so the cases the
# diagnostician reasons over and the ones shown in the UI are the same set.
#
# Without a floor the store returned the nearest five whatever their score, and
# "my car smells like vanilla" pulled five brake repairs at 0.21-0.33 into the
# prompt under the heading "similar past cases you may cite as evidence".
CASE_MATCH_MIN_SIMILARITY = env.float("CASE_MATCH_MIN_SIMILARITY", default=0.7)

# The second, much higher bar: at or above this a past diagnosis is served
# straight back and the diagnostician, parts and shops agents never run -- the
# whole point being that re-deriving an answer we already have costs three
# model calls and half a minute to arrive at the same place.
#
# Deliberately far above the evidence floor. Between the two a case is context
# for a fresh answer; only a near-identical description of the same problem on
# the same car is allowed to *be* the answer. carmed applies its own 0.80 floor
# on top, so lowering this below that has no effect.
CASE_SERVE_MIN_SIMILARITY = env.float("CASE_SERVE_MIN_SIMILARITY", default=0.95)

# --- LLM access (OpenRouter -> DeepSeek gateway, see /setup/README.md) ---
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY", default=env("DEEP_SEEK_API_KEY", default=""))
OPENROUTER_BASE_URL = env("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
OPENROUTER_MODEL = env("OPENROUTER_MODEL", default="deepseek/deepseek-v4-pro")

# How many prior DiagnosticMessage turns to hand the LLM as context for a reply.
DIAGNOSTIC_CHAT_HISTORY_LIMIT = env.int("DIAGNOSTIC_CHAT_HISTORY_LIMIT", default=10)

# --- The agentic layer (/carmed): see apps.diagnostics.agentic ---
# Cosine floor for serving a stored case as the answer instead of diagnosing
# afresh. carmed ships 0.80; cosine has no absolute meaning across embedding
# models, so this is applied to carmed.graph in apps.diagnostics.apps.ready()
# and must be recalibrated whenever EMBEDDING_MODEL_NAME changes -- on held-out
# examples, never on the evaluation set.
CARMED_CACHE_HIT_THRESHOLD = env.float("CARMED_CACHE_HIT_THRESHOLD", default=0.80)

# Force "do not drive" whenever the symptom text mentions brakes, steering,
# suspension or airbags. On by default here because CaseRecord carries no
# urgency column -- a cached answer would otherwise inherit carmed's neutral
# default, and understating a brake problem is the one error that hurts someone.
CARMED_SAFETY_FLOOR = env.bool("CARMED_SAFETY_FLOOR", default=True)

# Used for shop lookups when a request doesn't name a city.
CARMED_DEFAULT_CITY = env("CARMED_DEFAULT_CITY", default="Yerevan")

# How many past completed cases to pull per gate check (carmed default is 5).
CARMED_CASE_LIMIT = env.int("CARMED_CASE_LIMIT", default=5)
