#%%writefile deploy.py
import subprocess
import threading
import time
from pyngrok import ngrok
import os

NGROK_AUTH_TOKEN = "35WtIZpirdsJRc5i4LJRatmJ9JP_2kujDdW25KB7R6YyoiKhd"

def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8502", "--server.address", "0.0.0.0"])

def deploy_with_ngrok():
    print("🔐 Setting up ngrok...")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    print("🚀 Starting application...")
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    time.sleep(8)
    
    try:
        public_url = ngrok.connect(8502, bind_tls=True)
        print(f"✅ Application is running at: {public_url}")
        print("💡 Share this link to access the application!")
        
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("🛑 Shutting down...")
            ngrok.kill()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    deploy_with_ngrok()