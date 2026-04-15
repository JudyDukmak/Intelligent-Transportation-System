FROM cityflowproject/cityflow:latest

# Upgrade pip FIRST
RUN pip install --upgrade pip setuptools wheel

# Install small packages first
RUN pip install numpy==1.19.5 gym==0.17.3 matplotlib pandas

# Install torch from CPU wheel manually
RUN pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Then install SB3
RUN pip install stable-baselines3==1.3.0