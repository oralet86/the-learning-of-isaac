-- Minimal JSON encode/decode for Lua 5.4.
--
-- Vendored because neither vanilla Isaac Lua nor REPENTOGON documents a built-in
-- JSON codec (see IsaacDocs' Isaac class page — only SaveModData/LoadModData for
-- string persistence). Handles the plain-data subset needed for IPC: nested
-- tables/arrays, strings, numbers, booleans, and nil-as-absence.

---@class JSON
local JSON = {}

-- ---------------------------------------------------------------------------
-- Encode
-- ---------------------------------------------------------------------------

local ESCAPE_MAP = {
    ['"'] = '\\"',
    ["\\"] = "\\\\",
    ["\b"] = "\\b",
    ["\f"] = "\\f",
    ["\n"] = "\\n",
    ["\r"] = "\\r",
    ["\t"] = "\\t",
}

---@param s string
---@return string
local function encode_string(s)
    local out = { '"' }
    for i = 1, #s do
        local c = s:sub(i, i)
        out[#out + 1] = ESCAPE_MAP[c] or c
    end
    out[#out + 1] = '"'
    return table.concat(out)
end

---@param t table
---@return boolean
local function is_array(t)
    local n = 0
    for _ in pairs(t) do
        n = n + 1
    end
    if n == 0 then
        return true -- empty table encodes as [] per encode_value below
    end
    for i = 1, n do
        if t[i] == nil then
            return false
        end
    end
    return true
end

local encode_value

---@param t table
---@return string
local function encode_array(t)
    local parts = {}
    for i = 1, #t do
        parts[i] = encode_value(t[i])
    end
    return "[" .. table.concat(parts, ",") .. "]"
end

---@param t table
---@return string
local function encode_object(t)
    local parts = {}
    for k, v in pairs(t) do
        if type(k) ~= "string" then
            error("JSON.encode: object keys must be strings, got " .. type(k))
        end
        parts[#parts + 1] = encode_string(k) .. ":" .. encode_value(v)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

---@param v any
---@return string
encode_value = function(v)
    local t = type(v)
    if t == "nil" then
        return "null"
    elseif t == "boolean" then
        return v and "true" or "false"
    elseif t == "number" then
        if v ~= v or v == math.huge or v == -math.huge then
            error("JSON.encode: cannot encode non-finite number")
        end
        return tostring(v)
    elseif t == "string" then
        return encode_string(v)
    elseif t == "table" then
        if is_array(v) then
            return encode_array(v)
        end
        return encode_object(v)
    end
    error("JSON.encode: cannot encode value of type " .. t)
end

---@param value any
---@return string
function JSON.encode(value)
    return encode_value(value)
end

-- ---------------------------------------------------------------------------
-- Decode
-- ---------------------------------------------------------------------------

---@class JSONParser
---@field str string
---@field pos integer
local Parser = {}
Parser.__index = Parser

---@param str string
---@return JSONParser
local function new_parser(str)
    return setmetatable({ str = str, pos = 1 }, Parser)
end

function Parser:skip_whitespace()
    local _, stop = self.str:find("^%s*", self.pos)
    self.pos = stop + 1
end

---@return string
function Parser:peek()
    return self.str:sub(self.pos, self.pos)
end

---@return any
function Parser:parse_value()
    self:skip_whitespace()
    local c = self:peek()
    if c == "{" then
        return self:parse_object()
    elseif c == "[" then
        return self:parse_array()
    elseif c == '"' then
        return self:parse_string()
    elseif c == "t" or c == "f" then
        return self:parse_boolean()
    elseif c == "n" then
        self.pos = self.pos + 4 -- "null"
        return nil
    else
        return self:parse_number()
    end
end

---@return table
function Parser:parse_object()
    self.pos = self.pos + 1 -- '{'
    local obj = {}
    self:skip_whitespace()
    if self:peek() == "}" then
        self.pos = self.pos + 1
        return obj
    end
    while true do
        self:skip_whitespace()
        local key = self:parse_string()
        self:skip_whitespace()
        assert(self:peek() == ":", "JSON.decode: expected ':' at pos " .. self.pos)
        self.pos = self.pos + 1
        local value = self:parse_value()
        obj[key] = value
        self:skip_whitespace()
        local c = self:peek()
        if c == "," then
            self.pos = self.pos + 1
        elseif c == "}" then
            self.pos = self.pos + 1
            break
        else
            error("JSON.decode: expected ',' or '}' at pos " .. self.pos)
        end
    end
    return obj
end

---@return table
function Parser:parse_array()
    self.pos = self.pos + 1 -- '['
    local arr = {}
    self:skip_whitespace()
    if self:peek() == "]" then
        self.pos = self.pos + 1
        return arr
    end
    while true do
        arr[#arr + 1] = self:parse_value()
        self:skip_whitespace()
        local c = self:peek()
        if c == "," then
            self.pos = self.pos + 1
        elseif c == "]" then
            self.pos = self.pos + 1
            break
        else
            error("JSON.decode: expected ',' or ']' at pos " .. self.pos)
        end
    end
    return arr
end

local UNESCAPE_MAP = {
    ['"'] = '"',
    ["\\"] = "\\",
    ["/"] = "/",
    b = "\b",
    f = "\f",
    n = "\n",
    r = "\r",
    t = "\t",
}

---@return string
function Parser:parse_string()
    assert(self:peek() == '"', "JSON.decode: expected '\"' at pos " .. self.pos)
    self.pos = self.pos + 1
    local out = {}
    while true do
        local c = self.str:sub(self.pos, self.pos)
        if c == "" then
            error("JSON.decode: unterminated string")
        elseif c == '"' then
            self.pos = self.pos + 1
            break
        elseif c == "\\" then
            local nc = self.str:sub(self.pos + 1, self.pos + 1)
            out[#out + 1] = UNESCAPE_MAP[nc] or nc
            self.pos = self.pos + 2
        else
            out[#out + 1] = c
            self.pos = self.pos + 1
        end
    end
    return table.concat(out)
end

---@return boolean
function Parser:parse_boolean()
    if self.str:sub(self.pos, self.pos + 3) == "true" then
        self.pos = self.pos + 4
        return true
    elseif self.str:sub(self.pos, self.pos + 4) == "false" then
        self.pos = self.pos + 5
        return false
    end
    error("JSON.decode: invalid literal at pos " .. self.pos)
end

---@return number
function Parser:parse_number()
    local match = self.str:match("^-?%d+%.?%d*[eE]?[+-]?%d*", self.pos)
    if not match or match == "" then
        error("JSON.decode: invalid number at pos " .. self.pos)
    end
    self.pos = self.pos + #match
    return tonumber(match)
end

---@param str string
---@return any
function JSON.decode(str)
    local parser = new_parser(str)
    local value = parser:parse_value()
    return value
end

return JSON
