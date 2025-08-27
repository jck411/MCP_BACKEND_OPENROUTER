"""
OpenRouter Models MCP Server

Provides intelligent search, filtering, and statistics for OpenRouter AI models.
Caches model data locally with on-demand refresh capability.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import requests
from mcp.server.fastmcp import FastMCP

# Configure logging
logger = logging.getLogger(__name__)

# Create server instance
mcp = FastMCP("OpenRouter Models")

# Database configuration
DB_PATH = "chat_history.db"  # Reuse existing database
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


class ModelDatabase:
    """Handles OpenRouter model data storage and querying."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Initialize the models table if not already done."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            async with aiosqlite.connect(self.db_path) as db:
                # Enable optimizations (matching existing setup)
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=10000")
                await db.execute("PRAGMA temp_store=memory")
                
                # Create models table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS openrouter_models (
                        -- Primary identifiers
                        id TEXT PRIMARY KEY,
                        canonical_slug TEXT NOT NULL,
                        name TEXT NOT NULL,
                        
                        -- Basic info (indexed for common queries)
                        created INTEGER NOT NULL,
                        context_length INTEGER,
                        hugging_face_id TEXT,
                        description TEXT,
                        
                        -- Pricing (normalized to per-1M tokens for easy comparison)
                        price_prompt_per_1m REAL,
                        price_completion_per_1m REAL,
                        price_request_per_1m REAL,
                        price_image_per_1m REAL,
                        price_web_search_per_1m REAL,
                        price_internal_reasoning_per_1m REAL,
                        price_input_cache_read_per_1m REAL,
                        price_input_cache_write_per_1m REAL,
                        price_audio_per_1m REAL,
                        
                        -- Provider info (commonly queried)
                        is_moderated BOOLEAN,
                        max_completion_tokens INTEGER,
                        
                        -- Architecture (searchable)
                        tokenizer TEXT,
                        instruct_type TEXT,
                        
                        -- Complex data as JSON (for flexibility)
                        architecture_json TEXT,
                        pricing_raw_json TEXT,
                        top_provider_json TEXT,
                        supported_parameters_json TEXT,
                        
                        -- Metadata
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        raw_response_json TEXT
                    )
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_models_created ON openrouter_models(created)",
                    "CREATE INDEX IF NOT EXISTS idx_models_context_length ON openrouter_models(context_length)",
                    "CREATE INDEX IF NOT EXISTS idx_models_price_prompt ON openrouter_models(price_prompt_per_1m)",
                    "CREATE INDEX IF NOT EXISTS idx_models_price_completion ON openrouter_models(price_completion_per_1m)",
                    "CREATE INDEX IF NOT EXISTS idx_models_tokenizer ON openrouter_models(tokenizer)",
                    "CREATE INDEX IF NOT EXISTS idx_models_is_moderated ON openrouter_models(is_moderated)",
                    "CREATE INDEX IF NOT EXISTS idx_models_name ON openrouter_models(name)",
                ]
                
                for index_sql in indexes:
                    try:
                        await db.execute(index_sql)
                    except Exception as e:
                        # Skip index creation if column doesn't exist yet (will be created on first data load)
                        logger.debug(f"Skipping index creation: {e}")
                
                await db.commit()
            
            self._initialized = True
    
    def _convert_pricing_to_per_million(self, pricing: Dict[str, str]) -> Dict[str, Optional[float]]:
        """Convert pricing from per-token to per-1M tokens."""
        result = {}
        for key, value in pricing.items():
            try:
                if value and value != "0":
                    per_token = float(value)
                    per_million = per_token * 1_000_000
                    result[f"price_{key}_per_1m"] = per_million
                else:
                    result[f"price_{key}_per_1m"] = None
            except (ValueError, TypeError):
                result[f"price_{key}_per_1m"] = None
        return result
    
    async def refresh_models(self) -> Tuple[int, str]:
        """Fetch latest models from OpenRouter API and update database."""
        await self._ensure_initialized()
        
        try:
            logger.info("Fetching models from OpenRouter API...")
            response = requests.get(OPENROUTER_API_URL, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("data", [])
            
            if not models:
                return 0, "No models received from API"
            
            logger.info(f"Processing {len(models)} models...")
            
            async with aiosqlite.connect(self.db_path) as db:
                # Clear existing data
                await db.execute("DELETE FROM openrouter_models")
                
                # Insert new data
                for model in models:
                    # Extract and convert pricing
                    pricing = model.get("pricing", {})
                    pricing_per_1m = self._convert_pricing_to_per_million(pricing)
                    
                    # Extract provider info
                    top_provider = model.get("top_provider", {})
                    
                    # Extract architecture
                    architecture = model.get("architecture", {})
                    
                    # Prepare row data
                    row_data = {
                        "id": model.get("id"),
                        "canonical_slug": model.get("canonical_slug"),
                        "name": model.get("name"),
                        "created": model.get("created"),
                        "context_length": model.get("context_length"),
                        "hugging_face_id": model.get("hugging_face_id"),
                        "description": model.get("description"),
                        "is_moderated": top_provider.get("is_moderated"),
                        "max_completion_tokens": top_provider.get("max_completion_tokens"),
                        "tokenizer": architecture.get("tokenizer"),
                        "instruct_type": architecture.get("instruct_type"),
                        "architecture_json": json.dumps(architecture),
                        "pricing_raw_json": json.dumps(pricing),
                        "top_provider_json": json.dumps(top_provider),
                        "supported_parameters_json": json.dumps(model.get("supported_parameters", [])),
                        "last_updated": datetime.now(UTC).isoformat(),
                        "raw_response_json": json.dumps(model),
                        **pricing_per_1m
                    }
                    
                    # Insert row
                    columns = ", ".join(row_data.keys())
                    placeholders = ", ".join("?" * len(row_data))
                    
                    await db.execute(
                        f"INSERT INTO openrouter_models ({columns}) VALUES ({placeholders})",
                        list(row_data.values())
                    )
                
                await db.commit()
            
            logger.info(f"Successfully cached {len(models)} models")
            return len(models), f"Successfully cached {len(models)} models"
        
        except Exception as e:
            error_msg = f"Failed to refresh models: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about cached models."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            stats = {}
            
            # Basic counts
            async with db.execute("SELECT COUNT(*) as total FROM openrouter_models") as cursor:
                row = await cursor.fetchone()
                stats["total_models"] = row[0] if row else 0
            
            if stats["total_models"] == 0:
                return {"total_models": 0, "message": "No models cached. Run refresh_models first."}
            
            # Last updated
            async with db.execute("SELECT MAX(last_updated) as last_update FROM openrouter_models") as cursor:
                row = await cursor.fetchone()
                stats["last_updated"] = row[0] if row else None
            
            # Pricing statistics
            pricing_fields = [
                "price_prompt_per_1m", "price_completion_per_1m", "price_request_per_1m",
                "price_image_per_1m", "price_web_search_per_1m", "price_internal_reasoning_per_1m",
                "price_input_cache_read_per_1m", "price_input_cache_write_per_1m", "price_audio_per_1m"
            ]
            
            pricing_stats = {}
            for field in pricing_fields:
                async with db.execute(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT({field}) as non_null,
                        MIN({field}) as min_price,
                        MAX({field}) as max_price,
                        AVG({field}) as avg_price
                    FROM openrouter_models
                    WHERE {field} IS NOT NULL AND {field} > 0
                """) as cursor:
                    row = await cursor.fetchone()
                    if row and row[1] > 0:  # non_null count
                        pricing_stats[field.replace("price_", "").replace("_per_1m", "")] = {
                            "available_models": row[1],
                            "min_per_1m_tokens": round(row[2], 4) if row[2] else None,
                            "max_per_1m_tokens": round(row[3], 4) if row[3] else None,
                            "avg_per_1m_tokens": round(row[4], 4) if row[4] else None,
                        }
            
            stats["pricing"] = pricing_stats
            
            # Free models
            async with db.execute("""
                SELECT COUNT(*) FROM openrouter_models 
                WHERE (price_prompt_per_1m IS NULL OR price_prompt_per_1m = 0)
                AND (price_completion_per_1m IS NULL OR price_completion_per_1m = 0)
            """) as cursor:
                row = await cursor.fetchone()
                stats["free_models"] = row[0] if row else 0
            
            # Capabilities
            capabilities_stats = {}
            common_params = ["tools", "tool_choice", "reasoning", "structured_outputs", "response_format"]
            for param in common_params:
                async with db.execute(f"""
                    SELECT COUNT(*) FROM openrouter_models 
                    WHERE json_extract(supported_parameters_json, '$') LIKE '%{param}%'
                """) as cursor:
                    row = await cursor.fetchone()
                    capabilities_stats[param] = row[0] if row else 0
            
            stats["capabilities"] = capabilities_stats
            
            # Context lengths
            async with db.execute("""
                SELECT 
                    MIN(context_length) as min_context,
                    MAX(context_length) as max_context,
                    AVG(context_length) as avg_context
                FROM openrouter_models 
                WHERE context_length IS NOT NULL
            """) as cursor:
                row = await cursor.fetchone()
                if row:
                    stats["context_length"] = {
                        "min": row[0],
                        "max": row[1],
                        "avg": round(row[2]) if row[2] else None
                    }
            
            # Modalities
            async with db.execute("""
                SELECT DISTINCT json_extract(architecture_json, '$.input_modalities') as input_mods
                FROM openrouter_models 
                WHERE input_mods IS NOT NULL
            """) as cursor:
                rows = await cursor.fetchall()
                input_modalities = set()
                for row in rows:
                    try:
                        mods = json.loads(row[0])
                        input_modalities.update(mods)
                    except:
                        pass
                stats["input_modalities"] = sorted(input_modalities)
            
            # Tokenizers
            async with db.execute("""
                SELECT tokenizer, COUNT(*) as count 
                FROM openrouter_models 
                WHERE tokenizer IS NOT NULL 
                GROUP BY tokenizer 
                ORDER BY count DESC
            """) as cursor:
                rows = await cursor.fetchall()
                stats["tokenizers"] = {row[0]: row[1] for row in rows}
            
            return stats
    
    async def search_models(
        self,
        query: Optional[str] = None,
        max_prompt_price: Optional[float] = None,
        max_completion_price: Optional[float] = None,
        min_context_length: Optional[int] = None,
        has_tools: Optional[bool] = None,
        has_reasoning: Optional[bool] = None,
        free_only: bool = False,
        tokenizer: Optional[str] = None,
        input_modality: Optional[str] = None,
        output_modality: Optional[str] = None,
        unmoderated_only: bool = False,
        limit: int = 50,
        sort_by: str = "created",
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Search models with flexible filtering options."""
        await self._ensure_initialized()
        
        # Build WHERE clauses
        where_clauses = []
        params = []
        
        if query:
            where_clauses.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if max_prompt_price is not None:
            where_clauses.append("(price_prompt_per_1m IS NULL OR price_prompt_per_1m <= ?)")
            params.append(max_prompt_price)
        
        if max_completion_price is not None:
            where_clauses.append("(price_completion_per_1m IS NULL OR price_completion_per_1m <= ?)")
            params.append(max_completion_price)
        
        if min_context_length is not None:
            where_clauses.append("context_length >= ?")
            params.append(min_context_length)
        
        if has_tools:
            where_clauses.append("(json_extract(supported_parameters_json, '$') LIKE '%tools%' OR json_extract(supported_parameters_json, '$') LIKE '%tool_choice%')")
        
        if has_reasoning:
            where_clauses.append("json_extract(supported_parameters_json, '$') LIKE '%reasoning%'")
        
        if free_only:
            where_clauses.append("(price_prompt_per_1m IS NULL OR price_prompt_per_1m = 0) AND (price_completion_per_1m IS NULL OR price_completion_per_1m = 0)")
        
        if tokenizer:
            where_clauses.append("tokenizer = ?")
            params.append(tokenizer)
        
        if input_modality:
            where_clauses.append(f"json_extract(architecture_json, '$.input_modalities') LIKE '%{input_modality}%'")
        
        if output_modality:
            where_clauses.append(f"json_extract(architecture_json, '$.output_modalities') LIKE '%{output_modality}%'")
        
        if unmoderated_only:
            where_clauses.append("is_moderated = 0")
        
        # Build ORDER BY
        valid_sort_fields = {
            "created": "created",
            "name": "name",
            "context_length": "context_length",
            "prompt_price": "price_prompt_per_1m",
            "completion_price": "price_completion_per_1m"
        }
        
        sort_field = valid_sort_fields.get(sort_by, "created")
        sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        # Build final query
        base_query = "SELECT * FROM openrouter_models"
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        base_query += f" ORDER BY {sort_field} {sort_direction} LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(base_query, params) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    row_dict = dict(row)
                    # Parse JSON fields back to objects for response
                    try:
                        row_dict["architecture"] = json.loads(row_dict["architecture_json"])
                        row_dict["pricing_raw"] = json.loads(row_dict["pricing_raw_json"])
                        row_dict["top_provider"] = json.loads(row_dict["top_provider_json"])
                        row_dict["supported_parameters"] = json.loads(row_dict["supported_parameters_json"])
                    except:
                        pass
                    
                    # Clean up internal fields
                    for field in ["architecture_json", "pricing_raw_json", "top_provider_json", "supported_parameters_json", "raw_response_json"]:
                        row_dict.pop(field, None)
                    
                    results.append(row_dict)
                
                return results


# Global database instance
db = ModelDatabase()


class QueryProcessor:
    """Processes natural language queries and converts them to structured searches."""
    
    @staticmethod
    def parse_natural_query(query: str) -> Dict[str, Any]:
        """Parse natural language query into search parameters."""
        query_lower = query.lower()
        params = {}
        
        # Check if this is a structural query that shouldn't use text search
        is_structural_query = any(pattern in query_lower for pattern in [
            "models that", "models with", "show me", "find", "list",
            "output", "input", "support", "have", "can"
        ])
        
        # Free models
        if any(term in query_lower for term in ["free", "no cost", "zero cost", "$0"]):
            params["free_only"] = True
        
        # Tools/function calling
        if any(term in query_lower for term in ["tools", "function call", "tool call"]):
            params["has_tools"] = True
        
        # Reasoning
        if any(term in query_lower for term in ["reasoning", "think", "chain of thought"]):
            params["has_reasoning"] = True
        
        # Pricing thresholds
        if "under $" in query_lower or "less than $" in query_lower or "cheaper than $" in query_lower:
            import re
            price_match = re.search(r'(?:under|less than|cheaper than) \$?(\d+(?:\.\d+)?)', query_lower)
            if price_match:
                price = float(price_match.group(1))
                if "completion" in query_lower:
                    params["max_completion_price"] = price
                else:
                    params["max_prompt_price"] = price
        
        # Context length
        if any(term in query_lower for term in ["large context", "long context", "context"]):
            if "128k" in query_lower or "128000" in query_lower:
                params["min_context_length"] = 128000
            elif "100k" in query_lower or "100000" in query_lower:
                params["min_context_length"] = 100000
            elif "32k" in query_lower or "32000" in query_lower:
                params["min_context_length"] = 32000
        
        # Modalities
        if "image" in query_lower:
            if "input" in query_lower or "vision" in query_lower:
                params["input_modality"] = "image"
            elif "output" in query_lower or "generate" in query_lower:
                params["output_modality"] = "image"
        if "audio" in query_lower:
            if "input" in query_lower:
                params["input_modality"] = "audio"
            elif "output" in query_lower or "generate" in query_lower:
                params["output_modality"] = "audio"
        
        # Moderation
        if "unmoderated" in query_lower or "uncensored" in query_lower:
            params["unmoderated_only"] = True
        
        # Sorting
        if any(term in query_lower for term in ["newest", "latest", "recent"]):
            params["sort_by"] = "created"
            params["sort_order"] = "desc"
        elif any(term in query_lower for term in ["cheapest", "lowest price"]):
            if "completion" in query_lower:
                params["sort_by"] = "completion_price"
            else:
                params["sort_by"] = "prompt_price"
            params["sort_order"] = "asc"
        
        # If this is a structural query and we found parameters, don't use text search
        if is_structural_query and params:
            params["_skip_text_search"] = True
        
        return params


# MCP Server Tools using FastMCP decorators

@mcp.tool()
async def refresh_models() -> str:
    """Refresh the local cache of OpenRouter models from the API"""
    count, message = await db.refresh_models()
    return f"✅ {message}\n\nUse get_statistics to see details about the cached models."


@mcp.tool()
async def get_statistics() -> str:
    """Get comprehensive statistics about cached models including pricing ranges, capabilities, and counts"""
    stats = await db.get_statistics()
    
    if stats.get("total_models", 0) == 0:
        return "❌ No models cached yet. Run refresh_models first to fetch model data from OpenRouter."
    
    # Format statistics nicely
    result = f"📊 **OpenRouter Models Statistics**\n\n"
    result += f"**Total Models:** {stats['total_models']}\n"
    result += f"**Last Updated:** {stats.get('last_updated', 'Unknown')}\n"
    result += f"**Free Models:** {stats.get('free_models', 0)}\n\n"
    
    # Pricing statistics
    if "pricing" in stats:
        result += "**💰 Pricing Ranges (per 1M tokens):**\n"
        for pricing_type, data in stats["pricing"].items():
            result += f"• {pricing_type.title()}: ${data['min_per_1m_tokens']:.4f} - ${data['max_per_1m_tokens']:.4f} (avg: ${data['avg_per_1m_tokens']:.4f}) - {data['available_models']} models\n"
        result += "\n"
    
    # Capabilities
    if "capabilities" in stats:
        result += "**🛠️ Capabilities:**\n"
        for capability, count in stats["capabilities"].items():
            result += f"• {capability.replace('_', ' ').title()}: {count} models\n"
        result += "\n"
    
    # Context lengths
    if "context_length" in stats:
        ctx = stats["context_length"]
        result += f"**📏 Context Length:** {ctx['min']:,} - {ctx['max']:,} tokens (avg: {ctx['avg']:,})\n\n"
    
    # Modalities
    if "input_modalities" in stats:
        result += f"**🔄 Input Modalities:** {', '.join(stats['input_modalities'])}\n\n"
    
    # Tokenizers
    if "tokenizers" in stats:
        result += "**🔤 Tokenizers:**\n"
        for tokenizer, count in list(stats["tokenizers"].items())[:10]:  # Top 10
            result += f"• {tokenizer}: {count} models\n"
    
    return result


@mcp.tool()
async def search_models(
    query: str = None,
    max_prompt_price: float = None,
    max_completion_price: float = None,
    min_context_length: int = None,
    has_tools: bool = None,
    has_reasoning: bool = None,
    free_only: bool = False,
    tokenizer: str = None,
    input_modality: str = None,
    output_modality: str = None,
    unmoderated_only: bool = False,
    limit: int = 50,
    sort_by: str = "created",
    sort_order: str = "desc"
) -> str:
    """Search and filter models using natural language or structured parameters. 
    
    Examples: 'free models with tools', 'cheapest completion pricing', 'models under $1 per 1M tokens'
    
    Args:
        query: Natural language query or search terms
        max_prompt_price: Maximum price per 1M prompt tokens (USD)
        max_completion_price: Maximum price per 1M completion tokens (USD)
        min_context_length: Minimum context length in tokens
        has_tools: Filter for models that support tool/function calling
        has_reasoning: Filter for models that support reasoning traces
        free_only: Show only free models (no cost for prompt or completion)
        tokenizer: Filter by tokenizer type (GPT, Llama3, Claude, etc.)
        input_modality: Filter by input modality (text, image, audio, file)
        output_modality: Filter by output modality (text, image, audio)
        unmoderated_only: Show only unmoderated models
        limit: Maximum number of results to return (default: 50)
        sort_by: Field to sort by (created, name, context_length, prompt_price, completion_price)
        sort_order: Sort order (asc, desc)
    """
    # Prepare arguments
    arguments = {
        "query": query,
        "max_prompt_price": max_prompt_price,
        "max_completion_price": max_completion_price,
        "min_context_length": min_context_length,
        "has_tools": has_tools,
        "has_reasoning": has_reasoning,
        "free_only": free_only,
        "tokenizer": tokenizer,
        "input_modality": input_modality,
        "output_modality": output_modality,
        "unmoderated_only": unmoderated_only,
        "limit": limit,
        "sort_by": sort_by,
        "sort_order": sort_order
    }
    
    # Handle natural language query
    skip_text_search = False
    if query:
        natural_params = QueryProcessor.parse_natural_query(query)
        # Check if we should skip text search for structural queries
        skip_text_search = natural_params.pop("_skip_text_search", False)
        
        # Merge with explicit parameters (explicit takes precedence)
        for key, value in natural_params.items():
            if arguments.get(key) is None:
                arguments[key] = value
    
    # If this is a structural query, remove the text query to avoid conflicts
    if skip_text_search:
        arguments.pop("query", None)
    
    # Remove None values
    arguments = {k: v for k, v in arguments.items() if v is not None}
    
    # Perform search
    results = await db.search_models(**arguments)
    
    if not results:
        # Provide helpful guidance for empty results
        guidance = "❌ No models found matching your criteria.\n\n"
        guidance += "**Try these options:**\n"
        guidance += "• Use get_statistics to see available ranges\n"
        guidance += "• Use get_available_options to see filter values\n"
        guidance += "• Try broader criteria or different price ranges\n"
        guidance += "• Example queries: 'free models', 'models with tools under $2', 'large context unmoderated'\n"
        
        return guidance
    
    # Format results
    result = f"🔍 **Found {len(results)} models**\n\n"
    
    for i, model in enumerate(results[:10], 1):  # Show first 10 in detail
        result += f"**{i}. {model['name']}**\n"
        result += f"• ID: `{model['id']}`\n"
        result += f"• Context: {model['context_length']:,} tokens\n"
        
        # Pricing
        pricing_parts = []
        if model.get('price_prompt_per_1m'):
            pricing_parts.append(f"Prompt: ${model['price_prompt_per_1m']:.4f}/1M")
        if model.get('price_completion_per_1m'):
            pricing_parts.append(f"Completion: ${model['price_completion_per_1m']:.4f}/1M")
        
        if pricing_parts:
            result += f"• Pricing: {', '.join(pricing_parts)}\n"
        else:
            result += "• Pricing: Free\n"
        
        # Capabilities
        capabilities = []
        if model.get('supported_parameters'):
            params = model['supported_parameters']
            if 'tools' in params or 'tool_choice' in params:
                capabilities.append("Tools")
            if 'reasoning' in params:
                capabilities.append("Reasoning")
            if 'structured_outputs' in params:
                capabilities.append("Structured Output")
        
        if capabilities:
            result += f"• Capabilities: {', '.join(capabilities)}\n"
        
        # Modalities
        if model.get('architecture', {}).get('input_modalities'):
            modalities = model['architecture']['input_modalities']
            if len(modalities) > 1 or modalities[0] != 'text':
                result += f"• Input: {', '.join(modalities)}\n"
        
        result += f"• Tokenizer: {model.get('tokenizer', 'Unknown')}\n"
        result += "\n"
    
    if len(results) > 10:
        result += f"... and {len(results) - 10} more models.\n"
        result += "Use more specific filters to narrow down results.\n"
    
    return result


@mcp.tool()
async def get_available_options() -> str:
    """Get all available filter options and values for building queries"""
    stats = await db.get_statistics()
    
    if stats.get("total_models", 0) == 0:
        return "❌ No models cached yet. Run refresh_models first to fetch model data."
    
    result = "📋 **Available Filter Options**\n\n"
    
    # Pricing ranges
    if "pricing" in stats:
        result += "**💰 Pricing Filters (per 1M tokens):**\n"
        for pricing_type, data in stats["pricing"].items():
            result += f"• max_{pricing_type}_price: Range ${data['min_per_1m_tokens']:.4f} - ${data['max_per_1m_tokens']:.4f}\n"
        result += "\n"
    
    # Context length
    if "context_length" in stats:
        ctx = stats["context_length"]
        result += f"**📏 Context Length Filters:**\n"
        result += f"• min_context_length: Range {ctx['min']:,} - {ctx['max']:,} tokens\n"
        result += f"• Common values: 32000, 100000, 128000, 200000+\n\n"
    
    # Boolean filters
    result += "**✅ Boolean Filters:**\n"
    result += "• free_only: true/false (models with no cost)\n"
    result += "• has_tools: true/false (supports function calling)\n"
    result += "• has_reasoning: true/false (supports reasoning traces)\n"
    result += "• unmoderated_only: true/false (uncensored models)\n\n"
    
    # Tokenizers
    if "tokenizers" in stats:
        result += "**🔤 Tokenizer Options:**\n"
        tokenizers = list(stats["tokenizers"].keys())[:8]  # Top 8
        result += f"• tokenizer: {', '.join(tokenizers)}\n\n"
    
    # Input modalities
    if "input_modalities" in stats:
        result += f"**🔄 Input Modality Options:**\n"
        result += f"• input_modality: {', '.join(stats['input_modalities'])}\n"
        result += f"• output_modality: text, image (most models only output text)\n\n"
    
    # Sort options
    result += "**📊 Sort Options:**\n"
    result += "• sort_by: created, name, context_length, prompt_price, completion_price\n"
    result += "• sort_order: asc, desc\n\n"
    
    # Example queries
    result += "**💡 Example Natural Language Queries:**\n"
    result += "• 'free models with tools'\n"
    result += "• 'cheapest completion pricing'\n"
    result += "• 'models under $1 per 1M tokens'\n"
    result += "• 'large context unmoderated models'\n"
    result += "• 'image input models with reasoning'\n"
    result += "• 'newest GPT tokenizer models'\n"
    
    return result


if __name__ == "__main__":
    mcp.run()
