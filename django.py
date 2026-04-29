import sys
import os
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.urls import path

# 1. SETTINGS
# Django needs to know where it is and how to behave.
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="this-is-not-a-secret-for-dev-only",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["*"],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            "django.middleware.csrf.CsrfViewMiddleware",
            "django.middleware.clickjacking.XFrameOptionsMiddleware",
        ],
    )

# 2. VIEWS
# These are the logic functions that handle requests.
def home(request):
    return HttpResponse("""
        <h1>Django Single-File App</h1>
        <p>This is a fully functional Django view running from one script!</p>
        <a href="/hello/">Go to Hello Page</a>
    """)

def hello(request):
    return HttpResponse("<h2>Hello! This is another route.</h2><a href='/'>Back Home</a>")

# 3. URLS
# Mapping the URL paths to the functions above.
urlpatterns = [
    path("", home),
    path("hello/", hello),
]

# 4. EXECUTION
# This part handles the command line (like "runserver")
if __name__ == "__main__":
    execute_from_command_line(sys.argv)
