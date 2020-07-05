# Remember to change the corresponding paths in `config.yml`
TRAIN_PATH=train.txt
DEV_PATH=dev.txt
TEST_PATH=test.txt
VOCAB_FILE=dict.word
ENTITY_FREQUENCY=dict.ent_wf
ENTITY_TYPE=dict.enttype
RELATION_PATH=dict.relation
# Remove NA instances
awk -F '\t' '{if($9!="") print $0}' $DEV_PATH > $DEV_PATH.filtered
awk -F '\t' '{if($9!="") print $0}' $TEST_PATH > $TEST_PATH.filtered
# or "no_relation" for TACRED
# awk -F '\t' '{if($9!="no_relation") print $0}' $TEST_PATH > $TEST_PATH.filtered
# Build vocabulary
# Build Entity type vocab
cut -f4 $TRAIN_PATH | tr "-" "\n" | sort | uniq > $ENTITY_TYPE
# Build word vocab
cut -f7 -d$'\t' $TRAIN_PATH | tr " " "\n" | sort | uniq -c | sort -nr | sed 's/^[ ]\+\([0-9]\+\) /\1\t/g' | awk -F '\t' '{printf("%s\t%s\n", $2, $1)}' > $VOCAB_FILE
# Build entity vocab & frequency
awk -F '\t' '{printf("%s\n%s", $2, $3)}' $TRAIN_PATH | sort | uniq -c | sort -nr | sed 's/^[ ]\+\([0-9]\+\) /\1\t/g' | awk -F '\t' '{printf("%s\t%s\n", $2, $1)}' > $ENTITY_FREQUENCY
cut -f1 $ENTITY_FREQUENCY > $ENTITY_FREQUENCY
# Build relation vocab
cut -f9 $DEV_PATH | sort | uniq > $RELATION_PATH