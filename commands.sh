cd /path/to/Samples
mkdir -p force_rms

for dir in s[0-9][0-9][0-9]/; do
    s="${dir%/}"                      # e.g. s001
    src="$s/48-procs"
    for f in section_force.csv reaction_force_over_time.png frank_starling.png frank_starling.csv; do
        if [[ -f "$src/$f" ]]; then
            cp "$src/$f" "force_rms/${s}_$f"
        else
            echo "Missing: $src/$f"
        fi
    done
done