FROM supervisely/base-py-sdk:6.72.216

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git wget

RUN wget -O /root/mit_b5.pth https://github.com/supervisely-ecosystem/hrda/releases/download/v1.0.15/mit_b5.pth

RUN python -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1
RUN python -m pip install --no-cache-dir torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
RUN python -m pip install --no-cache-dir mmcv-full==1.3.7
RUN python -m pip install --no-cache-dir timm==0.9.2 cityscapesscripts==2.2.0 cycler==0.10.0
RUN python -m pip install --no-cache-dir gdown humanfriendly==9.2 kiwisolver==1.2.0
RUN python -m pip install --no-cache-dir kornia==0.5.8 Pillow==8.3.1 prettytable==2.1.0
RUN python -m pip install --no-cache-dir pyparsing==2.4.7 pytz==2020.1 PyYAML==5.4.1 yapf==0.31.0

RUN python -m pip install --no-cache-dir supervisely==6.73.486

LABEL python_sdk_version="6.73.486"