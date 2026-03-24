#!/bin/bash
# test_aityvan.sh — тест переводчика aityvan
# Использование: bash test_aityvan.sh [host:port]
# По умолчанию: localhost:7860

HOST="${1:-localhost:7860}"
URL="http://$HOST/translate"
PASS=0
FAIL=0

echo "=== Тест переводчика aityvan ==="
echo "URL: $URL"
echo ""

# Проверка доступности
if ! curl -s "http://$HOST/list-languages" > /dev/null 2>&1; then
    echo "✗ Сервер не отвечает на http://$HOST"
    echo "  Запустите сервер и попробуйте снова"
    exit 1
fi
echo "✓ Сервер работает"
echo ""

# Функция теста
test_translate() {
    local text="$1"
    local src="$2"
    local tgt="$3"
    local direction="$4"

    result=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$text\",\"src_lang\":\"$src\",\"tgt_lang\":\"$tgt\"}" \
        --max-time 30)

    translation=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('translation','ERROR'))" 2>/dev/null)

    if [ -z "$translation" ] || [ "$translation" = "ERROR" ]; then
        echo "  ✗ $direction: $text"
        echo "    Ошибка: $result"
        FAIL=$((FAIL+1))
    else
        echo "  ✓ $direction: $text"
        echo "    → $translation"
        PASS=$((PASS+1))
    fi
}

# --- Русский → Тувинский ---
echo "--- Русский → Тувинский ---"
test_translate "Привет!" "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
test_translate "Как дела?" "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
test_translate "Спасибо" "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
test_translate "До свидания" "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
test_translate "Я живу в Туве." "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
test_translate "Как вас зовут?" "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
test_translate "Сегодня хорошая погода." "rus_Cyrl" "tyv_Cyrl" "ru→tyv"
echo ""

# --- Тувинский → Русский ---
echo "--- Тувинский → Русский ---"
test_translate "Экии!" "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
test_translate "Кандыг сен?" "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
test_translate "Четтирдим" "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
test_translate "Байырлыг" "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
test_translate "Мен Тывада чурттап турар мен." "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
test_translate "Адыңар кымыл?" "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
test_translate "Бөгүн агаар-бойдус эки-дир." "tyv_Cyrl" "rus_Cyrl" "tyv→ru"
echo ""

# --- Проверка утечки памяти (10 запросов подряд) ---
echo "--- Проверка памяти (10 запросов) ---"
MEM_BEFORE=$(curl -s "http://$HOST/docs" > /dev/null 2>&1; ps aux | grep "[u]vicorn" | awk '{sum+=$6} END {print sum/1024}' 2>/dev/null)

for i in $(seq 1 10); do
    curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d '{"text":"Привет, как дела?","src_lang":"rus_Cyrl","tgt_lang":"tyv_Cyrl"}' \
        --max-time 30 > /dev/null
    echo -n "."
done
echo ""

MEM_AFTER=$(ps aux | grep "[u]vicorn" | awk '{sum+=$6} END {print sum/1024}' 2>/dev/null)
echo "  RAM до:  ${MEM_BEFORE:-?} MB"
echo "  RAM после: ${MEM_AFTER:-?} MB"
echo ""

# --- Итого ---
TOTAL=$((PASS+FAIL))
echo "=== Результат: $PASS/$TOTAL пройдено ==="
if [ "$FAIL" -gt 0 ]; then
    echo "⚠ $FAIL тестов не прошли"
    exit 1
else
    echo "✓ Все тесты пройдены!"
fi
