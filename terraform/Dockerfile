FROM debian:buster-slim
WORKDIR /tmp


# download binaries
RUN apt-get update \
    # && apt-get install -qq --no-install-recommends wget zip  git\
    && apt-get install -qq wget zip  git\
    && apt-get -y install openssh-client \
    && wget -qO consul.zip https://releases.hashicorp.com/consul/1.7.3/consul_1.7.3_linux_amd64.zip \
    && wget -qO terraform.zip https://releases.hashicorp.com/terraform/0.12.25/terraform_0.12.25_linux_amd64.zip \
    && wget -qO /usr/local/bin/cfssljson https://pkg.cfssl.org/R1.2/cfssljson_linux-amd64 \
    && wget -qO /usr/local/bin/cfssl https://pkg.cfssl.org/R1.2/cfssl_linux-amd64

# extract and install and cleanup
RUN unzip consul.zip -d /usr/local/bin/ \
    && unzip terraform.zip -d /usr/local/bin/ \
    && rm /tmp/* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# setup ssh keys

RUN chmod 755 /usr/local/bin/cfssl* \
    && mkdir ~/.ssh \
    && mkdir ~/.ssh/azure \
    && cd ~/.ssh/azure \
    && ssh-keygen -t rsa -f acs_key -q -P "" \
    && consul tls ca create \
    && mkdir nomad \
    && cd nomad \
    && cfssl print-defaults csr | cfssl gencert -initca - | cfssljson -bare azure-ca

WORKDIR /home
#COPY entrypoint.sh /tmp
ENTRYPOINT [ "/bin/bash" ]
#ENTRYPOINT [ "/tmp/entrypoint.sh" ]