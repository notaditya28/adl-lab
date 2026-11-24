#!/usr/bin/env bash
# Resumable, checksum-verified download of the PhysioNet multimodal gait dataset (CSV parts).
# Usage: bash download.sh csv   |   bash download.sh imu
set -u
cd "$(dirname "$0")/data"
B=https://physionet.org/files/multimodal-gait-dataset/1.0.0
[ -f SHA256SUMS.txt ] || curl -sSL --retry 10 -o SHA256SUMS.txt $B/SHA256SUMS.txt

local_path() {   # map dataset path -> local path
  case "$1" in
    Dataset/EEG_Data/EEG.csv/*) echo "EEG_csv/${1##*/}";;
    Dataset/EMG_Data/*.csv)     echo "EMG_Data/${1##*/}";;
    Dataset/FP_Data/*.csv)      echo "FP_Data/${1##*/}";;
    Dataset/IMU_Data/IMU.tar.gz) echo "IMU_Data/IMU.tar.gz";;
    Dataset/subject-info.csv)   echo "subject-info.csv";;
  esac
}

if [ "${1:-csv}" = imu ]; then
  f=IMU_Data/IMU.tar.gz; mkdir -p IMU_Data
  until curl -sSL --retry 10 --retry-all-errors --connect-timeout 20 -C - -o $f $B/Dataset/IMU_Data/IMU.tar.gz; do
    echo "retrying IMU at $(stat -c %s $f) bytes"; sleep 5; done
  echo "$(grep 'IMU.tar.gz' SHA256SUMS.txt | cut -d' ' -f1)  $f" | sha256sum -c - && echo IMU_OK
  exit
fi

for pass in 1 2 3 4 5; do
  : > todo.cfg; n=0
  grep -E "Dataset/(EMG_Data/.*csv|FP_Data/.*csv|EEG_Data/EEG.csv/.*csv|subject-info.csv)" SHA256SUMS.txt |
  while read -r sum path; do
    lp=$(local_path "$path"); [ -n "$lp" ] || continue
    mkdir -p "$(dirname "$lp")"
    if [ ! -f "$lp" ] || [ "$(sha256sum "$lp" | cut -d' ' -f1)" != "$sum" ]; then
      printf 'url = "%s/%s"\noutput = "%s"\n' "$B" "$path" "$lp" >> todo.cfg
    fi
  done
  n=$(grep -c '^url' todo.cfg)
  echo "pass $pass: $n files to fetch"
  [ "$n" -eq 0 ] && { echo CSV_OK; rm -f todo.cfg; exit 0; }
  curl -sSL --retry 10 --retry-all-errors --connect-timeout 20 --parallel --parallel-max 8 -K todo.cfg
  sleep 3
done
echo CSV_INCOMPLETE
