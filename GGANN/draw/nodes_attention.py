import numpy as np
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 7718 1
def get_attention_from_npy_attention():
    file = open("graph.npy",'rb')
    n_attention = np.load(file)
    [rows, cols] = n_attention.shape
    attentions = []
    for i in range(rows):
        for j in range(cols):
            attentions.append(n_attention[i][j])
    return attentions


def get_graphs(index):
    file = open("valid_graphs.json")
    json_file = json.load(file)
    return json_file[index]['node_features']


def get_symbols(index):
    symbols = ["TranslationUnitDecl",  "FunctionDecl", "==", "&",  "&&",  "!", "+", ">", "!=",  "-", "=", "<", "++", ",", ">=", "--",
    "||", "<=",  "%", "/",  "*", "%=",  "+=",  "~",  "/=", "^", "-=", "^=", "*=", ">>", "|",  "<<", ">>=",  "|=", "&=", "<<=", "->*",
    "CompoundStmt", "DeclStmt", "VarDecl", "WhileStmt", "BinaryOperator",  "CallExpr", "ImplicitCastExpr",
    "DeclRefExpr",  "StringLiteral", "UnaryOperator", "IntegerLiteral", "IfStmt",  "BreakStmt",  "ReturnStmt",
    "CXXMemberCallExpr","MemberExpr", "CXXOperatorCallExpr", "ParenExpr", "ArraySubscriptExpr", "CharacterLiteral","ForStmt",
    "TypedefDecl", "BuiltinType", "ContinueStmt", "ParmVarDecl", "CXXConstructExpr","ExprWithCleanups","MaterializeTemporaryExpr",
    "CXXBindTemporaryExpr","CXXFunctionalCastExpr", "TemplateSpecializationType","TemplateArgument",
    "RecordType","ClassTemplateSpecialization","ConditionalOperator","FunctionTemplateDecl","TemplateTypeParmDecl",
    "UnresolvedLookupExpr","CompoundAssignOperator","FloatingLiteral","UnaryExprOrTypeTraitExpr","CXXBoolLiteralExpr","DoStmt","FullComment",
    "ParagraphComment","TextComment","CXXRecordDecl","FieldDecl","CXXConstructorDecl", "CXXThisExpr", "CXXMethodDecl","CStyleCastExpr",
    "CXXDestructorDecl","InitListExpr","array","ImplicitValueInitExpr", "CXXNewExpr","LabelStmt",
    "GotoStmt","GNUNullExpr","NullStmt","InlineCommandComment","CXXDefaultArgExpr","LinkageSpecDecl","ElaboratedType","CXXRecord","EnumDecl","EnumConstantDecl",
    "FunctionProtoType",
    "TypedefType",
    "Typedef",
    "PointerType",
    "QualType",
    "ConstantArrayType",
    "RestrictAttr",
    "FormatAttr",
    "DeprecatedAttr",
    "NamespaceDecl",
    "original",
    "AccessSpecDecl",
    "FriendDecl",
    "CXXCtorInitializer",
    "NonNullAttr",
    "PureAttr",
    "ConstAttr",
    "CXXDeleteExpr",
    "EmptyDecl",
    "CXXStaticCastExpr",
    "PredefinedExpr",
    "NoThrowAttr",
    "EnumType",
    "Enum",
    "VisibilityAttr",
    "DependentNameType",
    "CXXUnresolvedConstructExpr",
    "TypeOfExprType",
    "CXXScalarValueInitExpr",
    "SwitchStmt",
    "CaseStmt",
    "DefaultStmt",
    "ModeAttr",
    "WarnUnusedResultAttr",
    "AsmLabelAttr",
    "ParenType",
    "Function",
    "NonTypeTemplateParmDecl",
    "ParenListExpr",
    "CXXPseudoDestructorExpr",
    "ClassTemplateDecl",
    "ClassTemplateSpecializationDecl",
    "DependentScopeDeclRefExpr",
    "TemplateTypeParmType",
    "TemplateTypeParm",
    "CXXTryStmt",
    "CXXCatchStmt",
    "CXXThrowExpr",
    "CXXDependentScopeMemberExpr",
    "CXXTemporaryObjectExpr",
    "TypeTraitExpr",
    "public",
    "CXXConstCastExpr",
    "SubstTemplateTypeParmType",
    "UnresolvedMemberExpr",
    "protected",
    "CXXConversionDecl",
    "ClassTemplatePartialSpecializationDecl",
    "InjectedClassNameType",
    "LValueReferenceType",
    "CXXReinterpretCastExpr",
    "UnusedAttr",
    "AlignedAttr",
    "ReturnsTwiceAttr",
    "AliasAttr",
    "WeakRefAttr",
    "AtomicExpr",
    "IndirectFieldDecl",
    "Field",
    "AbiTagAttr",
    "CXXDynamicCastExpr",
    "virtual",
    "FormatArgAttr",
    "SubstNonTypeTemplateParmExpr",
    "TemplateTemplateParmDecl",
    "CompoundLiteralExpr",
    "CXXTypeidExpr",
    "CXXDefaultInitExpr",
    "GCCAsmStmt",
    "PackExpansionExpr",
    "CXXForRangeStmt",
    "TypeAliasDecl",
    "ConstructorAttr",
    "DestructorAttr",
    "AlwaysInlineAttr"]
    return symbols[index-1]


def plot(attentions):

    new_attentions = []
    i = 0
    while i<26:
        s_index = i*4
        new_attentions.append([attentions[s_index], attentions[s_index+1], attentions[s_index+2], attentions[s_index+3]])
        i += 1
    x = np.array(new_attentions)
    f, ax1 = plt.subplots(figsize=(6,6))

    sns.heatmap(x, annot=True, ax=ax1)
    plt.show()


if __name__ == '__main__':
    attentions = get_attention_from_npy_attention()
    plot(attentions)
    print(get_symbols(42))
    print(get_symbols(43))
    print(get_symbols(44))
    print(get_symbols(45))
    print(get_symbols(47))
    print(get_symbols(48))
    print(get_symbols(49))
    print(get_symbols(38))
    print(get_symbols(39))
    print(get_symbols(40))


