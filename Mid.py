from flask import Flask
from werkzeug.wrappers import Response

class APIKeyMiddleware:
    def __init__(self, app, valid_api_key):
        self.app = app
        self.valid_api_key = valid_api_key

    def __call__(self, environ, start_response):
        # Check if the request method is OPTIONS
        if environ.get('REQUEST_METHOD') == 'OPTIONS':
            # Return a CORS-compliant response for preflight requests
            response = Response('', status=200)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'X-API-KEY, Content-Type'
            return response(environ, start_response)

        # Existing API key validation for non-OPTIONS requests
        api_key = environ.get('HTTP_X_API_KEY')
        if not api_key or api_key != self.valid_api_key:
            response = Response('Invalid or missing API key', status=403)
            return response(environ, start_response)

        return self.app(environ, start_response)