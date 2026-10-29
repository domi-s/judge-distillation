# Sample submissions

These examples are part of the dataset, adapted from InfoAs.

## 100 points

### Example 1

```cpp
#include <iostream>
 
int main() {
	int a,b;
    std::cin >> a >> b;
    std::cout << a+b;
}
```

### Example 2

```cpp
#include<iostream>

using namespace std;

int main()
    
{
    int a,b;
    cin>>a>>b;
    cout<<a+b;
    
}
```

## 0 points

### Example 1

This compiles but does not produce the required output.

```cpp
#include <iostream>

using namespace std;

int main()
{
    int x, y;
    return 0;
}
```

### Example 2

This compiles but never terminates normally.

```cpp
#include <iostream>

using namespace std;

int main()
{
    while(1);
    int x, y;
    cin>>x>>y;
    cout<<x + y;
}
```

## Compilation error (0 points)

### Example 1

```cpp
a
```

### Example 2

```cpp
s
```

## 20 points

### Example 1

```cpp
#include <iostream>
using namespace std;
int main()
{
 	cout<<9+12;
    
}
```

### Example 2

```cpp
#include <iostream>
using namespace std;
int main()
{
    int a,b;
    cin>>a,b;
    cout<<a+b;
    return 0;
}
```
