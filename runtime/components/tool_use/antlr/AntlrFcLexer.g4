lexer grammar AntlrFcLexer;

// Structural Characters
OPEN_BRACE : '{';
CLOSE_BRACE : '}';
OPEN_BRACKET : '[';
CLOSE_BRACKET : ']';
COMMA : ',';
COLON : ':';

// Literals
BOOLEAN : 'true' | 'false';
NULL_LITERAL : 'null';

// Number: Integer and floating-point, including exponents
NUMBER : '-'? INT ( FRAC | EXP )? | '-'? FRAC | '-'? EXP ;

fragment INT : '0' | [1-9] [0-9]*;
fragment FRAC : '.' [0-9]+;
fragment EXP : [eE] [+-]? [0-9]+;

ESCAPED_STRING : '<escape>' .*? '<escape>';

CALL : 'call';
ID : [a-zA-Z_] [a-zA-Z_0-9]*;

// Whitespace: Skipped
WS : [ \t\n\r]+ -> skip;
