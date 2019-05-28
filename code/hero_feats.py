import json
import numpy as np 
import h5py

#id: attack_type; roles;
#    base_armor; base_mr; base_attack_min; base_attack_max; base_str; base_agi; base_int
#    str_gain; agi_gain; int_gain; attack_range; projectile_speed; attack_rate; move_speed
#    turn_rate; cm_enabled

# these 4 is useless: base_health; base_health_regen; base_mana; base_mana_regen
# useless feature: base_armor

path = "../dota-2-prediction/hero_names.json"

class hero_attrs():
    def __init__(self, path):
        self.load_feats(path)
    @property
    def feats_list(self):
        return ["primary_attr", "attack_type", "base_armor", "base_mr", "base_attack_min", 
            "base_attack_max", "base_str", "base_agi", "base_int", "str_gain", "agi_gain",
            "int_gain", "attack_range", "projectile_speed", "attack_rate", "roles"]
    @property
    def attr_value_map(self):
        attr_vs = [["int","agi","str"], ["Melee","Ranged"], ["Initiator","Jungler","Disabler","Pusher","Durable","Escape","Nuker","Support","Carry"]]
        value_map = {}
        for l in attr_vs:
            for v in l:
                value_map[v] = l.index(v)
        #value_map = {v:l.index(v) for v in l for l in attr_vs}
        """return {"primary_attr": ["int","agi","str"], 
                "attack_type": ["Melee","Ranged"],
                "roles": ["Initiator","Jungler","Disabler","Pusher","Durable","Escape","Nuker","Support","Carry"]}"""
        return value_map
    def load_feats(self, path):
        f = open(path)
        data = json.load(f)
        f.close()

        self.id_feats_map = {}
        for name in data:
            self.id_feats_map[data[name]["id"]] = {feat_name: data[name][feat_name] for feat_name in self.feats_list}
        
        return self.id_feats_map
    def analyse(self):
        output_f = open("./analyse_hero_attrs.txt", "w")
        for feat_name in self.feats_list:
            output_f.write(feat_name + ":\n")
            feat_rg = [self.id_feats_map[i][feat_name] for i in self.id_feats_map]
            if type(feat_rg[0]) is int or type(feat_rg[0]) is float:
                feat_rg = [-9999 if v is None else v for v in feat_rg]
                max_v = max(feat_rg)
                min_v = min(feat_rg)
                mean_v = float(sum(feat_rg) / len(feat_rg))
                output_f.write("max: " + str(max_v) + " min: " + str(min_v) + " mean: " + str(mean_v) + "\n")
            elif type(feat_rg[0]) is str:
                feat_rg_set = set(feat_rg)
                diff_v = len(feat_rg_set)
                output_f.write(",".join([str(diff_v)] + list(feat_rg_set)) + "\n")
            elif type(feat_rg[0]) is list:
                feat_rg_set = set([v for l in feat_rg for v in l])
                output_f.write(",".join([str(len(feat_rg_set))] + list(feat_rg_set)) + "\n")
    def make_feat(self):
        feats = {}
        for hid in self.id_feats_map:
            feat_vector = np.zeros((15+9), dtype=float)
            for idx, feat_name in enumerate(self.feats_list):
                v = self.id_feats_map[hid][feat_name]
                if type(v) is str:
                    v = self.attr_value_map[v]
                    feat_vector[idx] = v
                elif type(v) is list:
                    onehot_v = [0]*9
                    for r in v:
                        onehot_v[self.attr_value_map[r]] = 1
                    feat_vector[idx:] = onehot_v
                else:
                    feat_vector[idx] = v
            feats[hid] = feat_vector
        #print(feats[1])
        return feats

if __name__ == "__main__":
    md = hero_attrs(path)
    #md.analyse()
    #print(len(md.feats_list))
    md.make_feat()