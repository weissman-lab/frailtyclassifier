#!/bin/bash
set -e
set -o pipefail


get_page() {
  if [[ $("$(curl -s -w '%{http_code}' http://www.google.com)" | read response) ]]; then
    echo 'passed'
    status=${response:${#response}<3?0:-3}
    page=${response:0:-3}
  else
    echo 'failed'
  fi
}

get_page
echo $status
echo $page