import os
import sys
from huggingface_hub import HfApi, create_repo

def deploy():
    print("🌿 ZenFlow Hugging Face Deployer")
    print("================================")
    
    # Get token
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Please enter your Hugging Face Write Token: ").strip()
    
    if not token:
        print("❌ Error: Hugging Face Token is required.")
        return
        
    username = "shikhasrivastava0574"
    space_name = input("Enter the name for your new Hugging Face Space (default: ZenFlow): ").strip()
    if not space_name:
        space_name = "ZenFlow"
    repo_id = f"{username}/{space_name}"
    
    print(f"Creating/Locating Hugging Face Space: {repo_id}...")
    try:
        # Create the space repository on Hugging Face if it doesn't exist
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            token=token,
            private=False,
            exist_ok=True
        )
        print(f"✅ Space {repo_id} initialized!")
    except Exception as e:
        print(f"❌ Error setting up Space repository: {e}")
        return
        
    print("Uploading project files to Hugging Face...")
    api = HfApi(token=token)
    
    # List of files to upload
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "Readme.md",
        "Dockerfile"
    ]
    
    # Check if files exist and upload
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            try:
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=repo_id,
                    repo_type="space"
                )
                print(f"✅ Uploaded {file}")
            except Exception as e:
                print(f"❌ Failed to upload {file}: {e}")
        else:
            print(f"⚠️ Warning: {file} not found locally, skipping.")
            
    # Upload data/pdfs if present
    pdf_folder = "data/pdfs"
    if os.path.exists(pdf_folder):
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                local_path = os.path.join(pdf_folder, file)
                repo_path = f"data/pdfs/{file}"
                print(f"Uploading {repo_path}...")
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="space"
                    )
                    print(f"✅ Uploaded {repo_path}")
                except Exception as e:
                    print(f"❌ Failed to upload {repo_path}: {e}")

    print("\n🎉 Deployment completed successfully!")
    print(f"You can view your deployed space at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy()
