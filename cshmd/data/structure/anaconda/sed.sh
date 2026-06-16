for i in *.yaml; do
#    sed -i '' 's/\[6, 6, 1\]/[8, 10, 1]/g' "$i"
    sed -i '' 's/\[8, 8, 1\]/[10, 12, 1]/g' "$i"
done
