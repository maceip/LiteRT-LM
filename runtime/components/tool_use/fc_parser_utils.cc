// Copyright 2025 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/tool_use/fc_parser_utils.h"

#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "ANTLRInputStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTreeWalker.h"
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "AntlrFcLexer.h"
#include "AntlrFcParser.h"
#include "AntlrFcParserBaseListener.h"
#include "runtime/components/tool_use/parser_common.h"

namespace litert::lm {

namespace {

constexpr absl::string_view kEscape = "<escape>";

absl::string_view StripEscapeTokens(absl::string_view text) {
  text = absl::StripPrefix(text, kEscape);
  text = absl::StripSuffix(text, kEscape);
  return text;
}

absl::StatusOr<nlohmann::ordered_json> ParseArray(
    antlr_fc_tool_call_parser::AntlrFcParser::ArrayContext* array_ctx);

absl::StatusOr<nlohmann::ordered_json> ParseObject(
    antlr_fc_tool_call_parser::AntlrFcParser::ObjectContext* object_ctx);

// Parses a value context into a nlohmann::ordered_json.
absl::StatusOr<nlohmann::ordered_json> ParseValue(
    antlr_fc_tool_call_parser::AntlrFcParser::ValueContext* value_ctx) {
  if (value_ctx == nullptr) {
    return nlohmann::ordered_json();
  }

  if (value_ctx->ESCAPED_STRING()) {
    return nlohmann::ordered_json(
        std::string(StripEscapeTokens(value_ctx->getText())));
  } else if (value_ctx->NUMBER()) {
    double double_value;
    if (!absl::SimpleAtod(value_ctx->getText(), &double_value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse number: ", value_ctx->getText()));
    }
    return nlohmann::ordered_json(double_value);
  } else if (value_ctx->object()) {
    return ParseObject(value_ctx->object());
  } else if (value_ctx->array()) {
    return ParseArray(value_ctx->array());
  } else if (value_ctx->BOOLEAN()) {
    return nlohmann::ordered_json(value_ctx->getText() == "true");
  } else if (value_ctx->NULL_LITERAL()) {
    return nlohmann::ordered_json(nullptr);
  } else {
    // This cannot happen if the grammar is correct.
    return absl::InternalError(
        absl::StrCat("Unhandled value type: ", value_ctx->getText()));
  }
}

// Parses an array context into a nlohmann::ordered_json array.
absl::StatusOr<nlohmann::ordered_json> ParseArray(
    antlr_fc_tool_call_parser::AntlrFcParser::ArrayContext* array_ctx) {
  nlohmann::ordered_json list_value = nlohmann::ordered_json::array();
  if (array_ctx == nullptr) {
    return list_value;
  }

  for (antlr_fc_tool_call_parser::AntlrFcParser::ValueContext* value :
       array_ctx->value()) {
    absl::StatusOr<nlohmann::ordered_json> parsed_value = ParseValue(value);
    if (!parsed_value.ok()) {
      return parsed_value.status();
    }
    list_value.push_back(std::move(parsed_value).value());
  }
  return list_value;
}

// Parses an object context into a nlohmann::ordered_json object.
absl::StatusOr<nlohmann::ordered_json> ParseObject(
    antlr_fc_tool_call_parser::AntlrFcParser::ObjectContext* object_ctx) {
  nlohmann::ordered_json object = nlohmann::ordered_json::object();
  if (object_ctx == nullptr) {
    return object;
  }

  for (antlr_fc_tool_call_parser::AntlrFcParser::PairContext* pair_ctx :
       object_ctx->pair()) {
    if (pair_ctx == nullptr || pair_ctx->ID() == nullptr ||
        pair_ctx->value() == nullptr) {
      // This cannot happen if the grammar is correct.
      return absl::InvalidArgumentError("Invalid pair in object.");
    }

    std::string key = pair_ctx->ID()->getText();
    if (key.empty()) {
      // This cannot happen if the grammar is correct.
      return absl::InvalidArgumentError("Object key is empty.");
    }

    // Ignore duplicate keys.
    if (object.contains(key)) {
      ABSL_LOG(INFO) << "Ignoring duplicate key: " << key;
      continue;
    }

    absl::StatusOr<nlohmann::ordered_json> parsed_value =
        ParseValue(pair_ctx->value());
    if (!parsed_value.ok()) {
      return absl::Status(parsed_value.status().code(),
                          absl::StrCat("Error parsing value for key '", key,
                                       "': ", parsed_value.status().message()));
    }
    object[key] = std::move(parsed_value).value();
  }
  return object;
}

class FcListener : public antlr_fc_tool_call_parser::AntlrFcParserBaseListener {
 public:
  void enterFunctionCall(
      antlr_fc_tool_call_parser::AntlrFcParser::FunctionCallContext* ctx)
      override {
    if (ctx == nullptr) {
      return;
    }
    nlohmann::ordered_json tool_call;
    if (ctx->ID() == nullptr) {
      tool_call["name"] = "";
    } else {
      tool_call["name"] = ctx->ID()->getText();
    }
    tool_call["arguments"] = nlohmann::ordered_json::object();
    if (ctx->ID() != nullptr) {
      absl::StatusOr<nlohmann::ordered_json> args = ParseObject(ctx->object());
      if (args.ok()) {
        tool_call["arguments"] = std::move(*args);
      } else {
        tool_call["arguments"] = nlohmann::ordered_json::object();
        status_ = args.status();
      }
    }
    tool_calls_.push_back(tool_call);
  }

  const nlohmann::ordered_json& tool_calls() const { return tool_calls_; }
  absl::Status status() const { return status_; }

 private:
  nlohmann::ordered_json tool_calls_ = nlohmann::ordered_json::array();
  absl::Status status_ = absl::OkStatus();
};

}  // namespace

absl::StatusOr<nlohmann::ordered_json> ParseFcExpression(
    absl::string_view text) {
  if (text.empty()) {
    return nlohmann::ordered_json::array();
  }

  antlr4::ANTLRInputStream input(std::string(text.begin(), text.end()));
  antlr_fc_tool_call_parser::AntlrFcLexer lexer(&input);
  lexer.removeErrorListeners();
  DefaultErrorListener lexer_error_listener;
  lexer.addErrorListener(&lexer_error_listener);

  antlr4::CommonTokenStream tokens(&lexer);
  tokens.fill();  // Consume all tokens from the lexer.

  if (!lexer_error_listener.status()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Lexer failed to tokenize input.", text));
  }
  antlr_fc_tool_call_parser::AntlrFcParser parser(&tokens);
  parser.removeErrorListeners();
  DefaultErrorListener parser_error_listener;
  parser.addErrorListener(&parser_error_listener);

  antlr_fc_tool_call_parser::AntlrFcParser::StartContext* start_ctx =
      parser.start();

  if (!parser_error_listener.status() || parser.getNumberOfSyntaxErrors() > 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse input.", text));
  }

  if (start_ctx == nullptr) {
    return absl::InvalidArgumentError("Parsing resulted in a null context.");
  }

  FcListener listener;
  antlr4::tree::ParseTreeWalker::DEFAULT.walk(&listener, start_ctx);

  if (!listener.status().ok()) {
    // Listener reported one or more errors.
    return listener.status();
  }

  return listener.tool_calls();
}

}  // namespace litert::lm
