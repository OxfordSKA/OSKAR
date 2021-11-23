# clang-tidy-to-junit

A little script that can convert Clang-Tidy output to a JUnit XML file.

Usage:

```bash
$ cat clang-tidy-output | ./clang-tidy-to-junit.py /path/to/repository >junit.xml
```

You have to specify the `/path/to/repository` in order for the script to shorten the filenames in the `junit.xml`.
