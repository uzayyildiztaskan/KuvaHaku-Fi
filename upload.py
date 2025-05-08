import os
#from kaggle_secrets import UserSecretsClient
from huggingface_hub import create_branch, delete_branch, login, notebook_login, get_token, whoami, HfApi, HfFolder


def upload_to_huggingface(trainer):

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', None) is not None:

        if UserSecretsClient().get_secret("HF_TOKEN") is not None:
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")

            HfFolder.save_token(hf_token)


        else:
            print('''
                When using Kaggle, make sure to use the secret key HF_TOKEN with a 'WRITE' token.
                This will prevent the need to login every time you run the script.
                Set your secrets with the secrets add-on on the top of the screen.
                ''')
            notebook_login(write_permission=True)

        None


    elif os.environ.get('COLAB_BACKEND_VERSION', None) is not None:

        None

    trainer.push_to_hub("uyildiztaskan/kuvahaku-fi")