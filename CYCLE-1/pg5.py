a, b, c = [int(input()) for _ in range(3)]
if a == 0:
    print(round(-c / b, 2))
else:
    d = b * b - 4 * a * c
    if d < 0:
        print('no roots')
    elif d == 0:
        print(round(-b / 2 / a, 2))
    else:
        print(round((-b - d ** 0.5) / 2 / a, 2))
        print(round((-b + d ** 0.5) / 2 / a, 2))