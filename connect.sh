#!/bin/bash


# IP of the EC2 instance
USERNAME=ubuntu
IP=51.20.85.138

# SSH key file
SSH_File=~/.ssh/aws-rene.pem


GIT_HUB_repo=https://github.com/joseph-elf/Rene


REMOTE_COMMANDS="
python3 --version;
git clone $GIT_HUB_repo;
ls -a;
grep -qxF 'export PATH=\"\$HOME/EC2_connect:\$PATH\"' ~/.bashrc || \
echo 'export PATH=\"\$HOME/EC2_connect:\$PATH\"' >> ~/.bashrc
source ~/.bashrc;
"



# Write the NEXT_PUBLIC_API_URL to .env.local
# ssh -i $SSH_File -t ubuntu@$IP

# python3 --version

ssh -i $SSH_File -t $USERNAME@$IP "$REMOTE_COMMANDS exec bash"


