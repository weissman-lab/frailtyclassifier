FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install pandas numpy spacy scispacy scikit-learn scipy configargparse gensim sqlalch\
emy pymssql pyyaml
RUN python -m spacy download en_core_web_sm
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_s\
ci_md-0.3.0.tar.gz
WORKDIR /usr/src/app
CMD ["_07_AL_CV.py"]
ENTRYPOINT ["python"]