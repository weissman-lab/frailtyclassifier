#!/bin/bash
set -e
set -o pipefail


get_pk() {
	local response
	while true
	do
		#check if endpoint is up
		if response="$(curl -s -w '%{http_code}' http://vault.service.consul:8200/v1/ssh/public_key)"; then
			status=${response:${#response}<3?0:-3}
			public_key=${response:0:-3}
			if [[ $status -ne 200 ]]; then
				sleep 1
				continue
			fi
			#check if pk is valid
			if [[ ! $public_key =~ .*ssh-rsa.* ]]; then
				sleep 1
				continue
			fi
			break
		else
			echo 'call to get pk failed'
		fi
	  continue
	done
}

echo "Run SSH setup for Vault"
get_pk

sudo echo "TrustedUserCAKeys /etc/ssh/trusted-user-ca-keys.pem" >> /etc/ssh/sshd_config
sudo curl -o /etc/ssh/trusted-user-ca-keys.pem http://vault.service.consul:8200/v1/ssh/public_key
sudo chmod 600 /etc/ssh/trusted-user-ca-keys.pem
sudo service sshd stop
sudo service sshd start
echo "SSH setup completed"
