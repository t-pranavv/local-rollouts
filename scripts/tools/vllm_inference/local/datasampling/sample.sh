#!/usr/bin/env bash
set -euo pipefail

#############################################
# Argument parsing
#############################################

# Defaults
DATAROOT=/datadisk/datasets/newseeds/ci/v1/contests/dedup
N_SAMPLES=100
SEED=42
EXTRA_ARGS=()
COMBINE=1
COMBINED_OUTPUT=code_contests_sample_200.jsonl
COMBINED_LIMIT=""
PYTHON_BIN=python
SAMPLE_SCRIPT=sample_seed.py
PATTERN_REGEX="*.jsonl"

print_usage() {
	cat <<'USAGE'
Usage: sample.sh [options]

Options:
	--dataroot PATH            Root directory to search for *.jsonl (default: $DATAROOT)
	--n-samples N              Number of samples per file (default: 100)
	--seed N                   Random seed (default: 42)
	--extra-arg ARG            Extra argument passed to sampling script
	--pattern REGEX            Posix-extended regex for files (overrides default '*.jsonl'). If it does not start with '/', it's auto-prefixed with '.*/'
	--python PATH              Python executable (default: python)
	--script FILE              Sampling Python script (default: sample_seed.py)
	--combined-output FILE     Output file for combined samples (default: combined_samples.jsonl)
	--combined-limit N         Limit total combined records (default: unlimited)
	--no-combine               Skip combining step
	--combine                  Force combine (default)
	--help                     Show this help and exit

All non-option arguments after a standalone -- are forwarded to the sampling script for every file.
USAGE
}

FORWARDED_AFTER_DASH_DASH=()
while [[ $# -gt 0 ]]; do
	case $1 in
		--dataroot)
			DATAROOT=$2; shift 2;;
		--n-samples)
			N_SAMPLES=$2; shift 2;;
		--seed)
			SEED=$2; shift 2;;
		--extra-arg)
			EXTRA_ARGS+=("$2"); shift 2;;
		--pattern)
			PATTERN_REGEX=$2; shift 2;;
		--python)
			PYTHON_BIN=$2; shift 2;;
		--script)
			SAMPLE_SCRIPT=$2; shift 2;;
		--combined-output)
			COMBINED_OUTPUT=$2; shift 2;;
		--combined-limit)
			COMBINED_LIMIT=$2; shift 2;;
		--no-combine)
			COMBINE=0; shift;;
		--combine)
			COMBINE=1; shift;;
		--help|-h)
			print_usage; exit 0;;
		--)
			shift; FORWARDED_AFTER_DASH_DASH+=("$@"); break;;
		*)
			echo "Unknown option: $1" >&2
			print_usage >&2
			exit 1;;
	esac
done

# Merge forwarded args (after --) into EXTRA_ARGS
if [[ ${#FORWARDED_AFTER_DASH_DASH[@]} -gt 0 ]]; then
	EXTRA_ARGS+=("${FORWARDED_AFTER_DASH_DASH[@]}")
fi

dataroot=$DATAROOT
echo "Configuration:" >&2
echo "  dataroot         = $dataroot" >&2
echo "  n_samples        = $N_SAMPLES" >&2
echo "  seed             = $SEED" >&2
echo "  combine?         = $COMBINE" >&2
echo "  combined_output  = $COMBINED_OUTPUT" >&2
echo "  combined_limit   = ${COMBINED_LIMIT:-none}" >&2
echo "  python_bin       = $PYTHON_BIN" >&2
echo "  sample_script    = $SAMPLE_SCRIPT" >&2
echo "  pattern_regex    = $PATTERN_REGEX" >&2
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	echo "  extra_args       = ${EXTRA_ARGS[*]}" >&2
fi

echo "Sampling all .jsonl files under: $dataroot" >&2

# Function to compute output path given input file
# Rules:
#  1. If path contains '/filtered/' replace that segment with '/sampled/'
#  2. Else if parent directory isn't 'sampled', create sibling dir 'sampled' and put file there
compute_output_path() {
	local in="$1"
	if [[ "$in" == *"/filtered/"* ]]; then
		echo "${in/\/filtered\//\/sampled\/}"  # bash pattern replace
	else
		local dir base parent sampled_dir
		dir="$(dirname -- "$in")"
		base="$(basename -- "$in")"
		parent="$(basename -- "$dir")"
		if [[ "$parent" == "sampled" ]]; then
			echo "$dir/$base"
		else
			sampled_dir="$dir/sampled"
			sampled_dir="$(realpath -m "$sampled_dir")"
			echo "$sampled_dir/$base"
		fi
	fi
}

shopt -s nullglob globstar

pattern=$PATTERN_REGEX
if [[ $pattern != /* ]]; then
	pattern=".*/$pattern"
fi
mapfile -t files < <(find "$dataroot" -regextype posix-extended -type f -regex "$pattern" | sort)

if [[ ${#files[@]} -eq 0 ]]; then
	echo "No .jsonl files found under $dataroot" >&2
	exit 0
fi

for f in "${files[@]}"; do
	out="$(compute_output_path "$f")"
	out_dir="$(dirname -- "$out")"
	mkdir -p "$out_dir"
	echo "[+] Sampling: $f -> $out" >&2
	set -x
	"$PYTHON_BIN" "$SAMPLE_SCRIPT" \
		--input "$f" \
		--output "$out" \
		--n "$N_SAMPLES" \
		--seed "$SEED" \
		"${EXTRA_ARGS[@]}"
	set +x
done

echo "Done. Generated samples for ${#files[@]} files." >&2

if [[ "$COMBINE" == "1" ]]; then
	echo "Combining sampled files into $COMBINED_OUTPUT" >&2
	combine_cmd=("$PYTHON_BIN" combine_samples.py --dataroot "$dataroot" --output "$COMBINED_OUTPUT")
	if [[ -n "$COMBINED_LIMIT" ]]; then
		combine_cmd+=(--limit "$COMBINED_LIMIT")
	fi
	set -x
	"${combine_cmd[@]}"
	set +x
fi