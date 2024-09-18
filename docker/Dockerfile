FROM supervisely/base-py-sdk:6.72.216

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git wget

RUN pip3 install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111 # buildkit
RUN pip3 install mmcv-full==1.3.7
RUN pip3 install timm cityscapesscripts==2.2.0 cycler==0.10.0
RUN pip3 install gdown humanfriendly==9.2 kiwisolver==1.2.0
RUN pip3 install kornia==0.5.8 Pillow==8.3.1 prettytable==2.1.0
RUN pip3 install pyparsing==2.4.7 pytz==2020.1 PyYAML==5.4.1 yapf==0.31.0 # buildkit

RUN pip3 install setuptools==69.5.1
RUN pip3 install supervisely==6.73.190

RUN wget -O /root/mit_b5.pth https://deci-pretrained-models.s3.amazonaws.com/mit_backbones/mit_b5.pth # Link not safe
