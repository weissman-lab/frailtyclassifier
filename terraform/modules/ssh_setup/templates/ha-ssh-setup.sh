#!/bin/bash
set -e
set -o pipefail

get_kv() { curl --header "X-Consul-Token: $1" -s "http://127.0.0.1:8500/v1/kv/service/vault/$2?raw"; }

while [[ ! $vault_ssh_ready =~ .*ssh-rsa.* ]]; do
  echo "Waiting for vault SSH"
  sleep 1
  vault_ssh_ready=$(curl -s "http://vault.service.consul:8200/v1/ssh/public_key")
  echo $vault_ssh_ready
done


CONSUL_HTTP_TOKEN=$(consul kv get service/consul/vault-token)
echo $CONSUL_HTTP_TOKEN

echo "Admin name is set to:"
echo $admin_name
if [ ! -z "$admin_name" ]
then
    echo $admin_name
    token=$(get_kv $CONSUL_HTTP_TOKEN root-token)
    echo "Token is:"
    echo $token
    echo "Generate a pair of key"
    ssh-keygen -f /home/$admin_name/.ssh/id_rsa -t rsa -N ""
    echo "Keys generated"
    sleep 2
    echo $(cat /home/$admin_name/.ssh/id_rsa.pub)
    echo "Sign public key with Vault"
    sudo echo '{"public_key":"'$(cat /home/$admin_name/.ssh/id_rsa.pub)'"}' > /tmp/payload.json
    curl --header "X-Vault-Token: $token" --request POST --data @/tmp/payload.json http://vault.service.consul:8200/v1/ssh/sign/key | jq '.data.signed_key' --raw-output > /home/$admin_name/.ssh/id_rsa-cert.pub
fi