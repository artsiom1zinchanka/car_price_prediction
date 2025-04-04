import os

path = os.environ.get('PROJECT_PATH', '.')
print(os.getcwd(), os.path.abspath('.'))