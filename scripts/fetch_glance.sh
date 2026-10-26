#!/bin/bash
# Fetch the GLANCE repo. GitHub is intermittently unreachable from this host,
# so try codeload first then a series of public mirrors.
set -u
cd /home/mcy/Projects/GLANCE || exit 1
LOG=/home/mcy/Projects/GLANCE/scripts/fetch_glance.log
exec > >(tee -a "$LOG") 2>&1
echo "=========== START $(date) ==========="

try() {
  name="$1"; url="$2"
  [ -s glance_repo.tar.gz ] && return 0
  echo "### [$name] $url"
  curl -sSL --connect-timeout 15 --max-time 1500 -o glance_repo.tar.gz "$url"
  sz=$(stat -c%s glance_repo.tar.gz 2>/dev/null || echo 0)
  echo "    size=$sz"
  if [ "$sz" -lt 5000 ]; then rm -f glance_repo.tar.gz; echo "    (too small / failed)"; fi
}

try "codeload-main"   "https://codeload.github.com/js-lee-AI/GLANCE/tar.gz/refs/heads/main"
try "codeload-master" "https://codeload.github.com/js-lee-AI/GLANCE/tar.gz/refs/heads/master"
try "ghfast"          "https://ghfast.top/https://github.com/js-lee-AI/GLANCE/archive/refs/heads/main.tar.gz"
try "gh-proxy"        "https://gh-proxy.com/https://github.com/js-lee-AI/GLANCE/archive/refs/heads/main.tar.gz"
try "ghproxy-net"     "https://ghproxy.net/https://github.com/js-lee-AI/GLANCE/archive/refs/heads/main.tar.gz"
try "moeyy"           "https://github.moeyy.xyz/https://github.com/js-lee-AI/GLANCE/archive/refs/heads/main.tar.gz"
try "gitclone"        "https://gitclone.com/github.com/js-lee-AI/GLANCE/archive/refs/heads/main.tar.gz"
try "kkgithub"        "https://raw.kkgithub.com/js-lee-AI/GLANCE/archive/refs/heads/main.tar.gz"

if [ -s glance_repo.tar.gz ]; then
  rm -rf GLANCE_extract && mkdir GLANCE_extract
  tar xzf glance_repo.tar.gz -C GLANCE_extract
  d=$(find GLANCE_extract -maxdepth 1 -mindepth 1 -type d | head -1)
  rm -rf GLANCE_repo && mv "$d" GLANCE_repo
  echo "### OK extracted -> GLANCE_repo"
  find GLANCE_repo -type f | head -80
else
  echo "### ALL SOURCES FAILED"
fi
echo "=========== END $(date) ==========="
