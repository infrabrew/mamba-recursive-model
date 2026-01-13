package utils

import (
    "encoding/json"
    "io/ioutil"
)

type Config struct {{
    Host string `json:"host"`
    Port int    `json:"port"`
    Debug bool `json:"debug"`
}}

func LoadConfig(path string) (*Config, error) {{
    data, err := ioutil.ReadFile(path)
    if err != nil {{
        return nil, err
    }}

    var config Config
    err = json.Unmarshal(data, &config)
    return &config, err
}}