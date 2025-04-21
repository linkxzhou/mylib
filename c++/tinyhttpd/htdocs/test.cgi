#!/bin/sh

echo "Content-Type: text/plain"
echo
echo "hello, world"
echo "QUERY_STRING:$QUERY_STRING"
echo "CONTENT_LENGTH:$CONTENT_LENGTH"
echo "REQUEST_METHOD:$REQUEST_METHOD"