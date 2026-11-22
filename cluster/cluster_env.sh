# Cluster-specific settings for the shell launchers; source, don't execute.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"
#
# Lookup order matches stern/cluster_env.py: a variable already in the
# environment wins, then cluster/site.conf (gitignored, real host names),
# then cluster/site.conf.example (placeholders).

_cluster_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_cluster_env_load() {
  local file="$1" line key value
  [[ -f "${file}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "${line}" == *=* ]] || continue
    key="${line%%=*}"
    key="${key//[[:space:]]/}"
    [[ -n "${!key+set}" ]] && continue
    value="$(eval "printf '%s' ${line#*=}")"
    export "${key}=${value}"
  done < "${file}"
}

_cluster_env_load "${_cluster_env_dir}/site.conf"
_cluster_env_load "${_cluster_env_dir}/site.conf.example"

if [[ -z "${STERN_TIERS_FILE:-}" ]]; then
  if [[ -f "${_cluster_env_dir}/tiers.tsv" ]]; then
    STERN_TIERS_FILE="${_cluster_env_dir}/tiers.tsv"
  else
    STERN_TIERS_FILE="${_cluster_env_dir}/tiers.example.tsv"
  fi
fi
export STERN_TIERS_FILE
export STERN_DATA_ROOT="${STERN_DATA_ROOT:-/Data/${USER}}"
