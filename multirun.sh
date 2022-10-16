#!/bin/bash
# 汎用型複数パラメータ実行スクリプト
# argument1：パラメータの分割数
# argument2：この実行は何番目の実行か(0以上argument1未満)

# Ctrl+Cで全プロセスを強制終了
function terminate() {
    exit
}
trap 'terminate' {1,2,3,15}

# 引数の数が合っているか確認
if [ $# -ne 2 ]; then
    echo "This script needs two arguments to terminate."
    echo "First argument is split number for multi-gpu."
    echo "Second argument is index number. Must be between 0 and ({first argument} - 1)."
    exit 1
fi

# argument2が正しい範囲に存在するか確認
if test $2 -ge $1 -o $2 -lt 0 ; then
    echo "Second argument is index number. Must be between 0 and ({first argument} - 1)."
    exit 1
fi

# パラメータの保存先
param_path="./parameters/*"
# パラメータの保存ディレクトリを全て取得しディレクトリ名のみを配列に格納
dirs=`find ${param_path} -maxdepth 0 -type f | sed -e "s/\.\/parameters\///g" | sed -e "s/\.yaml//g"`
#echo $dirs

# 目的のパラメータのみ実行
i=0
for dir in $dirs;   # とりあえず走査(比較するパラメータ数が膨大な場合は変更する必要はあるが数百程度ならそのままでも問題ない)
do
    # i%argument1 == argument2の場合のみ実行
    if test $2 -eq $((i%$1)); then
        python main.py --yaml_name $dir
    fi
    # iをインクリメント
    let i++
done
